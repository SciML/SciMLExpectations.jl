# DiffEqUncertainty.jl

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/JuliaDiffEq/DiffEqUncertainty.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/DiffEqUncertainty.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaDiffEq/DiffEqUncertainty.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaDiffEq/DiffEqUncertainty.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaDiffEq/DiffEqUncertainty.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaDiffEq/DiffEqUncertainty.jl?branch=master)

DiffEqUncertainty.jl is a component package in the DifferentialEquations ecosystem. It holds the
utilities for solving uncertainty quantification. This includes quantifying uncertainties due to either:

- The propagation of initial condition and parametric uncertainties through an ODE
- The finite approximation of numerical solutions of ODEs and PDEs (ProbInts)

## Initial Condition and Parameteric Uncertanties

### Example
Here, we wish to compute the expected value for the number prey in the Lotka-Volterra model at 10s with uncertainty in the second initial condition and last model parameter. We will solve the expectation using two different algorithms, `MonteCarlo` and `Koopman`.

```julia
function f!(du,u,p,t)
    du[1] = p[1]*u[1] - p[2]*u[1]*u[2] #prey
    du[2] = -p[3]*u[2] + p[4]*u[1]*u[2] #predator
end

tspan = (0.0,10.0)
u0 = [1.0;1.0]
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(f,u0,tspan,p)

u0_dist = [1.0, Uniform(0.8, 1.1)]
p_dist = [1.5,1.0,3.0,truncated(Normal(1.0,.1),.6, 1.4)]

g(sol) = sol[1,end]

expectation(g, prob, u0_dist, p_dist, MonteCarlo(), Tsit5(); trajectories = 100_000)
expectation(g, prob, u0_dist, p_dist, Koopman(), Tsit5())  
```

If we wish to compute the variance, or 2nd central moment, of this same observable, we can do so as

```julia
centralmoment(2, g, prob, u0_dist, p_dist, Koopman(), Tsit5())[2]  
```

See [SciMLTutorials.jl](https://github.com/SciML/SciMLTutorials.jl) for additional examples.

### Expectations
DiffEqUncertainty.jl provides algorithms for computing the expectation of an observable, or quantity of interest, $g$ of the states of a dynamical system as the system evolves in time, i.e.,

$$\mathbb{E}\left[g\left(X\right)\right]$$

These algorithms are applicable to ODEs with initial condition and/or parametric uncertainty. Process noise is not currently supported. 

You can compute the expectation by using the `expectation` function:

```julia
expectation(g, prob, u0, p, expalg, args...; kwargs...)
```

- `g`: A function for computing the observable from an ODE solution `g(sol)`
- `prob`: An `ODEProblem`
- `u0`: Initial conditions. This can include a mixture of `Real` and `ContinuousUnivariateDistribution` (from Distributions.jl) types, e.g. `u0=[2.0, Uniform(1.0,2.0), Normal(4.0,1.0)]`. This allows you to specify both uncertain and deterministic initial conditions
- `p`: ODE parameters. This also can include a mixture of `Real` and `ContinuousUnivariateDistribution` (from Distributions.jl) types.
- `expalg`: Expectation algorithm to use

#### Algorithms
The following algorithms are available:

- `MonteCarlo`: Provides a convenient wrapper to `EnsembleProblem` for computing expectations via Monte Carlo simulation. Requires setting `trajectories >1`. See the [DifferentialEquations.jl documentation](https://diffeq.sciml.ai/stable/features/ensemble/#) for additional details.
- `Koopman`: Leverages the Koopman operator to compute the expectation efficiently via quadrature methods. This capability is built on top of DifferntialEquations.jl and Quadrature.jl. See Quadrature.jl for additional options. 

#### Common Keyword Arguments for `Koopman`
- `quadalg`: Quadrature algorithm. See Quadrature.jl for available algorithms
- `maxiter`: Maximum number of allowable quadrature iterations
- `ireltol`: Relative tolerance for quadrature integration
- `iabstol`: Absolute tolerance for quadrature integration
- `nout`: Output size of observable `g`. Used to specify vector-valued expectations
- `batch`: The preferred number of points to batch. This allows user-side
  parallelization of the expectation. See Quadrature.jl for additional details

### Central Moments
These algorithms can also be used to compute higher order central moments via `centralmoments`. This function returns the central moments up to the requested number.

```julia
centralmoments(n, args...; kwargs...)
```

- `n`: highest-order central moment to be computed. `centralmoments` will return an `n` length array with central moments 1 through `n`
- `args` and `kwargs`: This function wraps `expectation`. See `expectation` for additional options.

## ProbInts
Users interested in using this functionality should check out the [DifferentialEquations.jl documentation](https://diffeq.sciml.ai/stable/analysis/uncertainty_quantification/#ProbInts-1).


