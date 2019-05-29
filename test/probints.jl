using DiffEqUncertainty, DiffEqBase, OrdinaryDiffEq, DiffEqProblemLibrary, DiffEqMonteCarlo
using Test

using ParameterizedFunctions
g = @ode_def_bare LorenzExample begin
  dx = σ*(y-x)
  dy = x*(ρ-z) - y
  dz = x*y - β*z
end σ ρ β
u0 = [1.0;0.0;0.0]
tspan = (0.0,10.0)
p = [10.0,28.0,8/3]
prob = ODEProblem(g,u0,tspan,p)

cb = ProbIntsUncertainty(1e4,5)
solve(prob,Tsit5())
monte_prob = MonteCarloProblem(prob)
sim = solve(monte_prob,Tsit5(),num_monte=10,callback=cb,adaptive=false,dt=1/10)

#using Plots; plotly(); plot(sim,vars=(0,1),linealpha=0.4)

fitz = @ode_def FitzhughNagumo begin
  dV = 3.0*(V - V^3/3 + R)
  dR = -(1/3.0)*(V -  0.2 - 0.2*R)
end
u0 = [-1.0;1.0]
tspan = (0.0,20.0)
prob = ODEProblem(fitz,u0,tspan)

cb = ProbIntsUncertainty(0.1,1)
sol = solve(prob,Euler(),dt=1/10)
monte_prob = MonteCarloProblem(prob)
sim = solve(monte_prob,Euler(),num_monte=100,callback=cb,adaptive=false,dt=1/10)

#using Plots; plotly(); plot(sim,vars=(0,1),linealpha=0.4)

cb = AdaptiveProbIntsUncertainty(5)
sol = solve(prob,Tsit5())
monte_prob = MonteCarloProblem(prob)
sim = solve(monte_prob,Tsit5(),num_monte=100,callback=cb,abstol=1e-3,reltol=1e-1)

#using Plots; plotly(); plot(sim,vars=(0,1),linealpha=0.4)
