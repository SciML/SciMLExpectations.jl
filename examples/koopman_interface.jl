using OrdinaryDiffEq, Distributions, DiffEqUncertainty, Test, Quadrature
using Plots; gr()

## Discussion on dirac
# https://github.com/JuliaStats/Distributions.jl/issues/1098
# https://github.com/JuliaStats/Distributions.jl/issues/1040 -> Dirac(x) = Normal(x,zero(x))
# https://github.com/JuliaStats/Distributions.jl/pull/861
include("dirac.jl")


function eom(du,u,p,t)
  @inbounds begin
    du[1] = dx = p[1]*u[1] - u[1]*u[2]
    du[2] = dy = -3*u[2] + u[1]*u[2]
  end
  nothing
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5]

prob = ODEProblem(eom,u0,tspan,p)
sol = solve(prob,Tsit5())
plot(sol)


cost(sol) = sum(max(x[1]-6,0) for x in sol.u)

# u0s = [Uniform(0.25,5.5),Dirac(3.0)]
# ps  = [Uniform(1.0, )]

u0s = [Uniform(0.25,5.5),Uniform(0.25,5.5)]
ps  = [Uniform(0.5,2.0)]
ps = [Dirac(1.5)]


sol = koopman_expectation(cost,u0s,ps,prob,Tsit5();iabstol=1e-3,ireltol=1e-3,maxiters=1000,saveat=0.1)
