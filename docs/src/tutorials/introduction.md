# An Introduction to Expectations via SciMLExpectations.jl

## System Model

First, let's consider the following linear model.

$$u' = p u$$

```@example introduction
f(u, p, t) = p .* u
```

We then wish to solve this model on the timespan `t=0.0` to `t=10.0`, with an initial condition `u0=10.0` and parameter `p=-0.3`. We can then set up the differential equations, solve, and plot as follows

```@example introduction
using OrdinaryDiffEq, Plots
u0 = [10.0]
p = [-0.3]
tspan = (0.0, 10.0)
prob = ODEProblem(f, u0, tspan, p)
sol = solve(prob, Tsit5())
plot(sol)
```

However, what if we wish to consider a random initial condition? Assume `u0` is distributed uniformly from `-10.0` to `10.0`, i.e.,

```@example introduction
using Distributions
u0_dist = [Uniform(-10.0, 10.0)]
```

We can then run a Monte Carlo simulation of 100,000 trajectories by solving an `EnsembleProblem`.

```@example introduction
prob_func(prob, i, repeat) = remake(prob, u0 = rand.(u0_dist))
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)

ensemblesol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories = 100000)
```

Plotting a summary of the trajectories produces

```@example introduction
summ = EnsembleSummary(ensemblesol)
plot(summ)
```

Given the ensemble solution, we can then compute the expectation of
a function $g\left(\text{sol},p\right)$ of the system state `u` at any time
in the timespan, e.g., the state itself at `t=4.0` as

```@example introduction
g(sol, p) = sol(4.0)
mean([g(sol, prob.p) for sol in ensemblesol])
```

Alternatively, SciMLExpectations.jl offers a convenient interface for this type of calculation,
using `ExpectationProblem`.

```@example introduction
using SciMLExpectations
gd = GenericDistribution(u0_dist...)
h(x, u, p) = x, p
sm = SystemMap(prob, Tsit5())
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, MonteCarlo(100000))
sol.u
```

` ExpectationProblem` takes
a `SystemMap`, which contains the `ODEProblem` that we are working with, and how to solve it.
The function of interest $g$, which maps the solution of the `ODEProblem`
to the quantities you want to take the expectation of, in this case the state after 4 seconds.
A `GenericDistribution` containing all the aspects of the dynamical system you are uncertain about,
in this example this is the initial condition.
The function $h$, which maps a realization of the uncertainty space to the initial conditions and parameters of the `ODEProblem`.
Here, only the initial conditions are uncertain, while the parameters are deterministic.
See further down for an example with both uncertain initial conditions and parameters.

The `ExpectationProblem` can then be solved using an `AbstractExpectationAlgorithm`.
Here we use `MonteCarlo()` to use the Monte Carlo algorithm.
Recall, that `u0_dist = [Uniform(-10.0,10.0)]`, while `p = [-0.3]`. From this specification, the expectation is solved as

$$\mathbb{E}\left[g\left(S\left(h\left(x,u_0,p\right)\right)\right)\vert x\sim \text{gd}\right]$$

where $Pf$ is the “push-forward” density of the initial joint pdf $f$ on initial conditions and parameters.

Alternatively, we could solve the same problem using the `Koopman()` algorithm.

```@example introduction
sol = solve(exprob, Koopman())
sol.u
```

Being that this system is linear, we can analytically compute the solution as a deterministic ODE with its initial condition set to the expectation of the initial condition, i.e.,

$$e^{pt}\mathbb{E}\left[u_0\right]$$

```@example introduction
exp(p[1] * 4.0) * mean(u0_dist[1])
```

We see that for this case, the `Koopman()` algorithm produces a more accurate solution than `MonteCarlo()`. Not only is it more accurate, it is also much faster

```@example introduction
@time solve(exprob, MonteCarlo(100000))
```

```@example introduction
@time solve(exprob, Koopman())
```

Changing the distribution, we arrive at

```@example introduction
u0_dist = [Uniform(0.0, 10.0)]
gd = GenericDistribution(u0_dist...)
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, MonteCarlo(100000))
sol.u
```

and

```@example introduction
u0_dist = [Uniform(0.0, 10.0)]
gd = GenericDistribution(u0_dist...)
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, Koopman())
sol.u
```

where the analytical solution is

```@example introduction
exp(p[1] * 4.0) * mean(u0_dist[1])
```

Note that the `Koopman()` algorithm doesn't currently support infinite or semi-infinite integration domains, where the integration domain is determined by the extrema of the given distributions. So, trying to use a `Normal` distribution will produce `NaN`

```@example introduction
u0_dist = [Normal(3.0, 2.0)]
gd = GenericDistribution(u0_dist...)
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, Koopman())
sol.u
```

Here, the analytical solution is

```@example introduction
exp(p[1] * 4.0) * mean(u0_dist[1])
```

Using a truncated distribution will alleviate this problem. However, there is another gotcha. If a large majority of the probability mass of the distribution exists in a small region in the support, then the adaptive methods used to solve the expectation can “miss”  the non-zero portions of the distribution and erroneously return 0.0.

```@example introduction
u0_dist = [truncated(Normal(3.0, 2.0), -1000, 1000)]
gd = GenericDistribution(u0_dist...)
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, Koopman())
sol.u
```

whereas truncating at $\pm 4\sigma$ produces the correct result

```@example introduction
u0_dist = [truncated(Normal(3.0, 2.0), -5, 11)]
gd = GenericDistribution(u0_dist...)
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, Koopman())
sol.u
```

If a large truncation is required, it is best practice to center the distribution on the truncated interval. This is because many of the underlying quadrature algorithms use the center of the interval as an evaluation point.

```@example introduction
u0_dist = [truncated(Normal(3.0, 2.0), 3 - 1000, 3 + 1000)]
gd = GenericDistribution(u0_dist...)
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, Koopman())
sol.u
```

## Vector-Valued Functions

`ExpectationProblem` can also handle vector-valued functions.

Here, we demonstrate this by computing the expectation of `u` at `t=4.0s` and `t=6.0s`

```@example introduction
g(sol, p) = [sol(4.0)[1], sol(6.0)[1]]
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, Koopman())
sol.u
```

with analytical solution

```@example introduction
exp.(p .* [4.0, 6.0]) * mean(u0_dist[1])
```

this can be used to compute the expectation at a range of times simultaneously

```@example introduction
saveat = tspan[1]:0.5:tspan[2]
g(sol, p) = sol[1, :]
prob = ODEProblem(f, u0, tspan, p, saveat = saveat)
sm = SystemMap(prob, Tsit5())
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, Koopman())
sol.u
```

We can then plot these values along with the analytical solution

```@example introduction
plot(t -> exp(p[1] * t) * mean(u0_dist[1]), tspan..., xlabel = "t", label = "analytical")
scatter!(collect(saveat), sol.u, marker = :o, label = nothing)
```

### Benefits of Using Vector-Valued Functions

In the above examples, we used vector-valued expectation calculations to compute the various expectations required. Alternatively, one could simply compute multiple scalar-valued expectations. However, in most cases it is more efficient to use the vector-valued form. This is especially true when the ODE to be solved is computationally expensive.

To demonstrate this, let's compute the expectation of $x$, $x^2$, and $x^3$ using both approaches while counting the number of times `g()` is evaluated. This is the same as the number of simulation runs required to arrive at the solution. First, consider the scalar-valued approach. Here, we follow the same method as before, but we add a counter to our function evaluation that stores the number of function calls for each expectation calculation to an array.

```@example introduction
function g(sol, p, power, counter)
    counter[power] += 1
    sol(4.0)[1]^power
end
counter = [0, 0, 0]
g(sol, p, power) = g(sol, p, power, counter)
g(sol, p) = g(sol, p, 1)
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, Koopman())
sol.u
```

```@example introduction
g(sol, p) = g(sol, p, 2)
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, Koopman())
sol.u
```

```@example introduction
g(sol, p) = g(sol, p, 3)
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, Koopman())
sol.u
```

```@example introduction
counter
```

Leading to a total of `j sum(counters)` function evaluations.

Now, let's compare this to the vector-valued approach

```@example introduction
function g(sol, p, counter)
    counter[1] += 1
    v = sol(4.0)[1]
    [v, v^2, v^3]
end
counter = [0]
g(sol, p) = g(sol, p, counter)
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, Koopman())
sol.u
```

```@example introduction
counter
```

This is `j round(counter[1]/sum(counters)*100,digits=2)`% the number of simulations required when using scalar-valued expectations. Note how the number of evaluations used in the vector-valued form is equivalent to the maximum number of evaluations for the 3 scalar-valued expectation calls.

## Higher-Order Moments

Leveraging this vector-valued capability, we can also efficiently compute higher-order central moments.

### Variance

The variance, or 2nd central moment, of a random variable $X$ is defined as

$$\mathrm{Var}\left(X\right)=\mathbb{E}\left[\left(X-\mu\right)^2\right]$$

where

$$\mu = \mathbb{E}\left[X\right]$$

The expression for the variance can be expanded to

$$\mathrm{Var}\left(X\right)=\mathbb{E}\left[X^2\right]-\mathbb{E}\left[X\right]^2$$

Using this, we define a function that returns the expectations of $X$ and $X^2$ as a vector-valued function and then compute the variance from these

```@example introduction
function g(sol, p)
    x = sol(4.0)[1]
    [x, x^2]
end
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, Koopman())
sol.u
mean_g = sol.u[1]
var_g = sol.u[2] - mean_g^2
```

For a linear system, we can propagate the variance analytically as

$e^{2pt}\mathrm{Var}\left(u_0\right)$

```@example introduction
exp(2 * p[1] * 4.0) * var(u0_dist[1])
```

This can be computed at multiple time instances as well

```@example introduction
saveat = tspan[1]:0.5:tspan[2]
g(sol, p) = [sol[1, :]; sol[1, :] .^ 2]
prob = ODEProblem(f, u0, tspan, p, saveat = saveat)
sm = SystemMap(prob, Tsit5())
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, Koopman())
sol.u

μ = sol.u[1:length(saveat)]
σ = sqrt.(sol.u[(length(saveat) + 1):end] - μ .^ 2)

plot(t -> exp(p[1] * t) * mean(u0_dist[1]), tspan...,
    ribbon = t -> -sqrt(exp(2 * p[1] * t) * var(u0_dist[1])),
    label = "Analytical Mean, 1 std bounds")
scatter!(collect(saveat), μ, marker = :x, yerror = σ, c = :black,
    label = "Koopman Mean, 1 std bounds")
```

### Skewness

A similar approach can be used to compute skewness

```@example introduction
function g(sol, p)
    x = sol(4.0)[1]
    [x, x^2, x^3]
end
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, Koopman())
sol.u
mean_g = sol.u[1]
var_g = sol.u[2] - mean_g^2
skew_g = (sol.u[3] - 3.0 * mean_g * var_g - mean_g^3) / var_g^(3 / 2)
```

As the system is linear, we expect the skewness to be unchanged from the initial distribution. Because the distribution is a truncated Normal distribution centered on the mean, the true skewness is `0.0`.

## Batch-Mode

It is also possible to solve the various simulations in parallel by using the `batch` kwarg and a batch-mode supported quadrature algorithm via the `quadalg` kwarg. To view the list of batch compatible quadrature algorithms, refer to [Integrals.jl](https://docs.sciml.ai/Integrals/stable/). Note: Batch-mode operation is built on top of DifferentialEquation.jl's `EnsembleProblem`. See the [EnsembleProblem documentation](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/) for additional options.

The default quadrature algorithm to solve `ExpectationProblem` does not support batch-mode evaluation. So, we first load dependencies for additional quadrature algorithms

```@example introduction
using Cuba
```

We then solve our expectation as before, using a `batch=10` multi-thread parallelization via `EnsembleThreads()` of Cuba's SUAVE algorithm. We also introduce additional uncertainty in the model parameter.

```@example introduction
u0_dist = truncated(Normal(3.0, 2.0), -5, 11)
p_dist = truncated(Normal(-0.7, 0.1), -1, 0)
gd = GenericDistribution(u0_dist, p_dist)
g(sol, p) = sol(6.0)[1]
h(x, u, p) = [x[1]], [x[2]]
sm = SystemMap(prob, Tsit5(), EnsembleThreads())
exprob = ExpectationProblem(sm, g, h, gd)
# batchmode = EnsembleThreads() #where to pass this?
sol = solve(exprob, Koopman(), batch = 10, quadalg = CubaSUAVE())
sol.u
```

Now, let's compare the performance of the batch and non-batch modes

```@example introduction
@time solve(exprob, Koopman(), batch = 10, quadalg = CubaSUAVE())
```

```@example introduction
solve(exprob, Koopman(), quadalg = CubaSUAVE())
@time solve(exprob, Koopman(), quadalg = CubaSUAVE())
```

It is also possible to parallelize across the GPU. However, one must be careful of the limitations of ensemble solutions with the GPU. Please refer to [DiffEqGPU.jl](https://github.com/SciML/DiffEqGPU.jl) for details.

Here we load `DiffEqGPU` and modify our problem to use Float32 and to put the ODE in the required GPU form

!!! note
    

Switch `EnsembleCPUArray()` to `EnsembleGPUArray()` to make this example work on your GPU.
Currently, these docs are built on a machine without a GPU.

```julia
using DiffEqGPU

function f(du, u, p, t)
    @inbounds begin
        du[1] = p[1] * u[1]
    end
    nothing
end

u0 = Float32[10.0]
p = Float32[-0.3]
tspan = (0.0f0, 10.0f0)
prob = ODEProblem(f, u0, tspan, p)

g(sol, p) = sol(6.0)[1]

u0_dist = truncated(Normal(3.0f0, 2.0f0), -5.0f0, 11.0f0)
p_dist = truncated(Normal(-0.7f0, 0.1f0), -1.0f0, 0.0f0)
gd = GenericDistribution(u0_dist, p_dist)
g(sol, p) = sol(6.0f0)[1]
h(x, u, p) = [x[1]], [x[2]]
prob = ODEProblem(f, u0, tspan, p)
sm = SystemMap(prob, Tsit5(), EnsembleCPUArray())
exprob = ExpectationProblem(sm, g, h, gd)
sol = solve(exprob, Koopman(), batch = 10, quadalg = CubaSUAVE())
sol.u
```

The performance gains realized by leveraging batch GPU processing is problem-dependent. In this case, the number of batch evaluations required to overcome the overhead of using the GPU exceeds the number of simulations required to converge to the quadrature solution.
