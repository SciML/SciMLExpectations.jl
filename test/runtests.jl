using DiffEqUncertainty, DiffEqBase, OrdinaryDiffEq, DiffEqProblemLibrary, DiffEqMonteCarlo
using Base.Test

using ParameterizedFunctions
g = @ode_def_bare LorenzExample begin
  dx = σ*(y-x)
  dy = x*(ρ-z) - y
  dz = x*y - β*z
end σ=>10.0 ρ=>28.0 β=(8/3)
u0 = [1.0;0.0;0.0]
tspan = (0.0,10.0)
prob = ODEProblem(g,u0,tspan)

cb = ProbIntsUncertainty(1e4,5)
solve(prob,Tsit5())
sim = monte_carlo_simulation(prob,Tsit5(),num_monte=10,callback=cb,adaptive=false,dt=1/10)

using Plots; plot(sim,vars=(0,1),linealpha=0.4)
