# SciMLExpectations.jl: Expectated Values of Simulations and Uncertainty Quantification

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/SciMLExpectations/stable/)

[![codecov](https://codecov.io/gh/SciML/SciMLExpectations.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/SciMLExpectations.jl)
[![Build Status](https://github.com/SciML/SciMLExpectations.jl/workflows/CI/badge.svg)](https://github.com/SciML/SciMLExpectations.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

#### This package is still under heavy construction. Use at your own risk!

SciMLExpectations.jl is a package for quantifying the uncertainties of simulations by
calculating the expectations of observables with respect to input uncertainties. Its goal
is to make it fast and easy to compute solution moments in a differentiable way in order
to enable fast optimization under uncertainty.

## Tutorials and Documentation

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/SciMLExpectations/stable/). Use the
[in-development documentation](https://docs.sciml.ai/SciMLExpectations/dev/) for the version of
the documentation, which contains the unreleased features.

### Example

```julia
using SciMLExpectations, OrdinaryDiffEq, Distributions, Cubature

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

sm = SystemMap(prob, Tsit5(), save_everystep = false)

analytical = (exp(A * tspan[end]) * [mean(d) for d in u0s_dist])
analytical
```

```
julia> analytical
2-element Vector{Float64}:
  1.5433991194037804
 -1.120209038276938
```

```julia
g(sol, p) = sol[:, end]
exprob = ExpectationProblem(sm, g, cov, gd)
sol = solve(exprob, Koopman(); quadalg = CubatureJLh(),
    ireltol = 1e-3, iabstol = 1e-3)
sol.u # Expectation of the states 1 and 2 at the final time point
```

```
2-element Vector{Float64}:
  1.5433860531082695
 -1.1201922503747408
```

# Approximate error on the expectation

sol.resid
#=
2-element Vector{Float64}:
7.193424502016654e-5
5.2074632876847327e-5
=#

```
```
