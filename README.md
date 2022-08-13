# SciMLExpectations.jl: Expectated Values of Simulations and Uncertainty Quantification

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://github.com/SciML/SciMLExpectations.jl/workflows/CI/badge.svg)](https://github.com/SciML/SciMLExpectations.jl/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/JuliaDiffEq/SciMLExpectations.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaDiffEq/SciMLExpectations.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaDiffEq/SciMLExpectations.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaDiffEq/SciMLExpectations.jl?branch=master)

SciMLExpectations.jl is a package for quantifying the uncertainties of simulations by
calculating the expectations of observables with respect to input uncertainties. Its goal
is to make it fast and easy to compute solution moments in a differentiable way in order
to enable fast optimization under uncertainty.

### Example

```julia
using SciMLExpectations, OrdinaryDiffEq, Distributions, Integrals,
      IntegralsCubature, ComponentArrays, Random

function eom!(du, u, p, t, A)
    du .= A * u
end

u0 = [1.0, 1.0]
tspan = (0.0, 3.0)
p = [1.0; 2.0]
A = [0.0 1.0; -p[1] -p[2]]
prob = ODEProblem((du, u, p, t) -> eom!(du, u, p, t, A), u0, tspan, p)
u0s_dist = (Uniform(1, 10), truncated(Normal(3.0, 1), 0.0, 6.0))
gd = GenericDistribution(u0s_dist...)
cov(x, u, p) = x, p

sm = SystemMap(prob, Tsit5(), save_everystep=false)

analytical = (exp(A * tspan[end]) * [mean(d) for d in u0s_dist])

g(sol, p) = sol[:, end]
exprob = ExpectationProblem(sm, g, cov, gd; nout=length(u0))
sol = solve(exprob, Koopman(); quadalg=CubatureJLh(),
    ireltol=1e-3, iabstol=1e-3)

# Expectation of the states 1 and 2 at the final time point
sol.u

#=
2-element Vector{Float64}:
  1.5433860531082695
 -1.1201922503747408
=#

# Approximate error on the expectation
sol.resid
#=
2-element Vector{Float64}:
 7.193424502016654e-5
 5.2074632876847327e-5
=#
```
