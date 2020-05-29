using OrdinaryDiffEq, Distributions, DiffEqBase,
      DiffEqUncertainty, Test, Quadrature, Cubature

include("../examples/dirac.jl")

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
cost(sol) = sum(max(x[1]-6,0) for x in sol.u)

u0s = (Truncated(Normal(2.875, 0.875),0.25, 5.5) ,Truncated(Normal(2.875, 0.875),0.25, 5.5) )
ps  = (Truncated(Normal(1.25, 0.25),0.5, 2.0),)


@time sol = koopman_expectation(cost,u0s,ps,prob,Tsit5();iabstol=1e-3,ireltol=1e-3,maxiters=2000,saveat=0.1)
c1, e1 = sol.u, sol.resid

# batch
@time sol = koopman_expectation(cost,u0s,ps,prob,Tsit5(),EnsembleThreads();quadalg=CubatureJLh(),
                         batch=1000,iabstol=1e-3,ireltol=1e-3,
                         maxiters=2000,saveat=0.1)
c2, e2 = sol.u, sol.resid
@test abs(c1 - c2) < 0.1

# MC
@time c3 = montecarlo_expectation(cost,u0s,ps,prob,Tsit5(),EnsembleThreads();trajectories=100000,saveat=0.1)
@test abs(c1 - c3) < 0.1

###### New Interface vs old
@time sol = koopman_expectation2(cost,u0s,ps,prob,Tsit5();quadalg=CubatureJLh(),
                         iabstol=1e-3,ireltol=1e-3,
                         maxiters=2000,saveat=0.1)
c4, e4 = sol.u, sol.resid
@test abs(c1 - c4) < 0.1

# batch
@time sol = koopman_expectation2(cost,u0s,ps,prob,Tsit5(),EnsembleThreads();quadalg=CubatureJLh(),
                         batch=1000,iabstol=1e-3,ireltol=1e-3,
                         maxiters=2000,saveat=0.1)
c5, e5 = sol.u, sol.resid
@test abs(c1 - c5) < 0.1


###########



@testset "Deterministic vs Uncertain IC" begin
    u2 = [Truncated(Normal(2.875, 0.875),0.25, 5.5), Uniform(2.8749,2.8751), Truncated(Normal(2.8749,0.01),2,3), 2.875, Dirac(2.875)]
    ps  = [Truncated(Normal(1.25, 0.25),0.5, 2.0) ]
    for i in 1:length(u2)
        u0s = [Truncated(Normal(2.875, 0.875),0.25, 5.5) ,u2[i]]
        sol = koopman_expectation2(cost,u0s,ps,prob,Tsit5();quadalg=CubatureJLh(),
                                 batch = 0, iabstol=1e-3,ireltol=1e-3,
                                 maxiters=2000,saveat=0.1)
        c = sol.u
        sol = koopman_expectation2(cost,u0s,ps,prob,Tsit5(),EnsembleThreads();quadalg=CubatureJLh(),
                                 batch = 10, iabstol=1e-3,ireltol=1e-3,
                                 maxiters=2000,saveat=0.1)
        c_batch = sol.u
        c_mc = montecarlo_expectation(cost,u0s,ps,prob,Tsit5(),EnsembleThreads();trajectories=100000,saveat=0.1)
        @show c, c_batch, c_mc
        @test abs(c-c_mc) <0.1
        @test abs(c-c_batch) <0.1
    end
end


### Tuple vs Array
using BenchmarkTools
u0s = (Truncated(Normal(2.875, 0.875),0.25, 5.5) ,Truncated(Normal(2.875, 0.875),0.25, 5.5) )
ps  = (Truncated(Normal(1.25, 0.25),0.5, 2.0),)

@btime koopman_expectation2(cost,u0s,ps,prob,Tsit5();#quadalg=CubatureJLh(),
                         iabstol=1e-3,ireltol=1e-3,
                         maxiters=2000,saveat=0.1)
@show sol.u

u0s = [Truncated(Normal(2.875, 0.875),0.25, 5.5) ,Truncated(Normal(2.875, 0.875),0.25, 5.5) ]
ps  = [Truncated(Normal(1.25, 0.25),0.5, 2.0),]

@btime sol = koopman_expectation2(cost,u0s,ps,prob,Tsit5();#quadalg=CubatureJLh(),
                         iabstol=1e-3,ireltol=1e-3,
                         maxiters=2000,saveat=0.1)
@show sol.u
