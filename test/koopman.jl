using OrdinaryDiffEq, Distributions,
      DiffEqUncertainty, Test, Quadrature, Cubature


function f(du,u,p,t)
  @inbounds begin
    du[1] = dx = p[1]*u[1] - u[1]*u[2]
    du[2] = dy = -3*u[2] + u[1]*u[2]
  end
  nothing
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5]
prob = ODEProblem(f,u0,tspan,p)
sol = solve(remake(prob,u0=u0),Tsit5())
cost(sol) = sum(max(x[1]-6,0) for x in sol.u)
u0s = [Uniform(0.25,5.5),Uniform(0.25,5.5)]
ps  = [Uniform(0.5,2.0)]
sol = koopman_expectation(cost,u0s,ps,prob,Tsit5();iabstol=1e-3,ireltol=1e-3,maxiters=1000,saveat=0.1)
c1, e1 = sol.u, sol.resid
@time sol = koopman_expectation(cost,u0s,ps,prob,Tsit5(),EnsembleThreads();quadalg=CubatureJLh(),
                         batch=1000,iabstol=1e-3,ireltol=1e-3,
                         maxiters=2000,saveat=0.1)
c2, e2 = sol.u, sol.resid
@test abs(c1 - c2) < 0.1

@time c3 = montecarlo_expectation(cost,u0s,ps,prob,Tsit5(),EnsembleThreads();trajectories=100000,saveat=0.1)
@test abs(c1 - c3) < 0.1

##########
## Compare for all states/params random
begin
  u0 = [1.0;1.0]
  tspan = (0.0,10.0)
  p = [1.5]
  prob = ODEProblem(f,u0,tspan,p)
  u0s = [Uniform(0.25,5.5),Uniform(0.25,5.5)]
  ps  = [Uniform(0.5,2.0)]
  sol = expectation(cost,u0s,ps,prob,Tsit5();iabstol=1e-3,ireltol=1e-3,maxiters=1000,saveat=0.1)
  c4, e4 = sol.u, sol.resid
  @test c4 ≈ c1
end

## compare for subset of states random
begin
  u0 = [1.0;1.0]
  tspan = (0.0,10.0)
  p = [1.5]
  prob = ODEProblem(f,u0,tspan,p)

  u0s = [Uniform(0.25,5.5),Uniform( (1.0 .+ [-1e-8, 1e-8])...)]   # Old interface hacked by using very tight distribution for 2nd state
  ps  = [Uniform(0.5,2.0)]
  sol = koopman_expectation(cost,u0s,ps,prob,Tsit5();iabstol=1e-6,ireltol=1e-6,maxiters=1000,saveat=0.1)
  c5, e5 = sol.u, sol.resid

  u0s = [Uniform(0.25,5.5), 1.0]
  ps  = [Uniform(0.5,2.0)]
  prob = ODEProblem(f,[1.0,1.0],tspan,p)
  sol = expectation(cost,u0s,ps,prob,Tsit5();iabstol=1e-6,ireltol=1e-6,maxiters=1000,saveat=0.1)
  c6, e6 = sol.u, sol.resid

  @test abs(c6 - c5) < 0.02
end

## Compare with only random states
begin
  u0 = [1.0;1.0]
  tspan = (0.0,10.0)
  p = [1.5]
  prob = ODEProblem(f,u0,tspan,p)

  u0s = [Uniform(0.25,5.5),Uniform(0.25,5.5)]
  ps  = [Uniform((1.5 .+ [-1e-8, 1e-8])...)]  # Old interface hacked by using very tight distribution for 2nd state
  sol = koopman_expectation(cost,u0s,ps,prob,Tsit5();iabstol=1e-6,ireltol=1e-6,maxiters=1000,saveat=0.1)
  c7, e7 = sol.u, sol.resid

  u0s = [Uniform(0.25,5.5), Uniform(0.25,5.5)]
  ps  = [1.5]
  prob = ODEProblem(f,[1.0,1.0],tspan,p)
  sol = expectation(cost,u0s,ps,prob,Tsit5();iabstol=1e-6,ireltol=1e-6,maxiters=1000,saveat=0.1)
  c8, e8 = sol.u, sol.resid

  @test abs(c7 - c8) < 0.02
end
