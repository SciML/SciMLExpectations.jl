# Optimization Under Uncertainty with SciMLExpectations.jl

This tutorial gives and overview of how to leverage the efficient Koopman expectation method from SciMLExpectations to perform optimization under uncertainty. We demonstrate this by using a bouncing ball model with an uncertain model parameter. We also demonstrate its application to problems with probabilistic constraints, in particular a special class of constraints called chance constraints.

## System Model
First lets consider a 2D bouncing ball, where the states are the horizontal position $x$, horizontal velocity $\dot{x}$, vertical position $y$, and vertical velocity $\dot{y}$. This model has two system parameters, acceleration due to gravity and coefficient of restitution (models energy loss when the ball impacts the ground). We can simulate such a system using `ContinuousCallback` as

```@example control
using OrdinaryDiffEq, Plots

function ball!(du,u,p,t)
    du[1] = u[2]
    du[2] = 0.0
    du[3] = u[4]
    du[4] = -p[1]
end

ground_condition(u,t,integrator) = u[3]
ground_affect!(integrator) = integrator.u[4] = -integrator.p[2] * integrator.u[4]
ground_cb = ContinuousCallback(ground_condition, ground_affect!)

u0 = [0.0,2.0,50.0,0.0]
tspan = (0.0,50.0)
p = [9.807, 0.9]

prob = ODEProblem(ball!,u0,tspan,p)
sol = solve(prob,Tsit5(),callback=ground_cb)
plot(sol, vars=(1,3), label = nothing, xlabel="x", ylabel="y")
plot!(background_color = :transparent)
```

For this particular problem, we wish to measure the impact distance from a point $y=25$ on a wall at $x=25$. So, we introduce an additional callback that terminates the simulation on wall impact.

```@example control
stop_condition(u,t,integrator) = u[1] - 25.0
stop_cb = ContinuousCallback(stop_condition, terminate!)
cbs = CallbackSet(ground_cb, stop_cb)

tspan = (0.0, 1500.0)
prob = ODEProblem(ball!,u0,tspan,p)
sol = solve(prob,Tsit5(),callback=cbs)
plot(sol, vars=(1,3), label = nothing, xlabel="x", ylabel="y")
```

To help visualize this problem, we plot as follows, where the star indicates a desired impace location
```@example control
rectangle(xc, yc, w, h) = Shape(xc .+ [-w,w,w,-w]./2.0, yc .+ [-h,-h,h,h]./2.0)

begin
    plot(sol, vars=(1,3), label=nothing, lw = 3, c=:black)
    xlabel!("x [m]")
    ylabel!("y [m]")
    plot!(rectangle(27.5, 25, 5, 50), c=:red, label = nothing)
    scatter!([25],[25],marker=:star, ms=10, label = nothing,c=:green)
    ylims!(0.0,50.0)
end
```

## Considering Uncertainty
We now wish to introduce uncertainty in `p[2]`, the coefficient of restitution. This is defined via a continuous univiate distribution from Distributions.jl. We can then run a Monte Carlo simulation of 100 trajectories via the `EnsembleProblem` interface.

```@example control
using Distributions

cor_dist = truncated(Normal(0.9, 0.02), 0.9-3*0.02, 1.0)
trajectories = 100

prob_func(prob,i,repeat) = remake(prob, p = [p[1], rand(cor_dist)])
ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)
ensemblesol = solve(ensemble_prob,Tsit5(),EnsembleThreads(),trajectories=trajectories, callback=cbs)

begin # plot
    plot(ensemblesol, vars = (1,3), lw=1)
    xlabel!("x [m]")
    ylabel!("y [m]")
    plot!(rectangle(27.5, 25, 5, 50), c=:red, label = nothing)
    scatter!([25],[25],marker=:star, ms=10, label = nothing, c=:green)
    plot!(sol, vars=(1,3), label=nothing, lw = 3, c=:black, ls=:dash)
    xlims!(0.0,27.5)
end
```

Here, we plot the first 350 Monte Carlo simulations along with the trajectory corrresponding to the mean of the distribution (dashed line).

We now wish to compute the expected squared impact distance from the star. This is called an "observation" of our system or an "observable" of interest.

We define this observable as

```@example control
obs(sol,p) = abs2(sol[3,end]-25)
```

With the observable defined, we can compute the expected squared miss distance from our Monte Carlo simulation results as

```@example control
mean_ensemble = mean([obs(sol,p) for sol in ensemblesol])
```

Alternatively, we can use the `Koopman()` algorithm in SciMLExpectations.jl to compute this expectation much more efficiently as

```@example control
using SciMLExpectations
gd = GenericDistribution(cor_dist)
h(x, u, p) = u, [p[1];x[1]]
sm = SystemMap(prob, Tsit5(),callback=cbs)
exprob = ExpectationProblem(sm, obs, h, gd; nout=1)
sol = solve(exprob, Koopman(),ireltol = 1e-5)
sol.u
```

## Optimization Under Uncertainty
We now wish to optimize the initial position ($x_0,y_0$) and horizontal velocity ($\dot{x}_0$) of the system to minimize the expected squared miss distance from the star, where $x_0\in\left[-100,0\right]$, $y_0\in\left[1,3\right]$, and $\dot{x}_0\in\left[10,50\right]$. We will demonstrate this using a gradient-based optimization approach from NLopt.jl using `ForwardDiff.jl` AD through the expectation calculation.

First, we load the required packages and define our loss function

```@example control
using NLopt, SciMLSensitivity, ForwardDiff

make_u0(θ) = [θ[1],θ[2],θ[3], 0.0]

function 𝔼_loss(θ)
    prob = ODEProblem(ball!,make_u0(θ),tspan,p)
    sm = SystemMap(prob, Tsit5(),callback=cbs)
    exprob = ExpectationProblem(sm, obs, h, gd; nout=1)
    sol = solve(exprob, Koopman(),ireltol = 1e-5)
    sol.u
end
```

NLopt requires that this loss function return the loss as above, but also do an inplace update of the gradient. So, we wrap this function to put it in the form required by NLopt.

```@example control
function 𝔼_loss_nlopt(θ,∇)
    length(∇) > 0 ? ForwardDiff.gradient!(∇, 𝔼_loss,θ) : nothing
    𝔼_loss(θ)
end
```

We then optimize using the [Method of Moving Asymptotes](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#mma-method-of-moving-asymptotes-and-ccsa) algorithm (`:LD_MMA`)


```@example control
opt = Opt(:LD_MMA, 3)
opt.lower_bounds = [-100.0,1.0, 10.0]
opt.upper_bounds = [0.0,3.0, 50.0]
opt.xtol_rel = 1e-3
opt.min_objective = 𝔼_loss_nlopt
(minf,minx,ret) = NLopt.optimize(opt, [-1.0, 2.0, 50.0])
```

Let's now visualize 100 Monte Carlo simulations

```@example control
ensembleprob = EnsembleProblem(remake(prob,u0 = make_u0(minx)),prob_func=prob_func)
ensemblesol = solve(ensembleprob,Tsit5(),EnsembleThreads(), trajectories=100, callback=cbs)

begin
    plot(ensemblesol, vars = (1,3), lw=1,alpha=0.1)
    plot!(solve(remake(prob, u0=make_u0(minx)),Tsit5(), callback=cbs),
            vars=(1,3),label = nothing, c=:black, lw=3,ls=:dash)
    xlabel!("x [m]")
    ylabel!("y [m]")
    plot!(rectangle(27.5, 25, 5, 50), c=:red, label = nothing)
    scatter!([25],[25],marker=:star, ms=10, label = nothing,c=:green)
    ylims!(0.0,50.0)
    xlims!(minx[1], 27.5)
end
```

Looks pretty good! But, how long did it take? Let's benchmark.

```@example control
@time NLopt.optimize(opt, [-1.0, 2.0, 50.0])
```
Not bad for bound constrained optimization under uncertainty of a hybrid system!

## Probabilistic Constraints

With this approach we can also consider probabilistic constraints. Let us now consider a wall at $x=20$ with height 25.

```@example control
constraint = [20.0, 25.0]
begin
    plot(rectangle(27.5, 25, 5, 50), c=:red, label = nothing)
    xlabel!("x [m]")
    ylabel!("y [m]")
    plot!([constraint[1], constraint[1]],[0.0,constraint[2]], lw=5, c=:black, label=nothing)
    scatter!([25],[25],marker=:star, ms=10, label = nothing,c=:green)
    ylims!(0.0,50.0)
    xlims!(minx[1], 27.5)
end
```

We now wish to minimize the same loss function as before, but introduce an inequality constraint such that the solution must have less than a 1% chance of colliding with the wall at $x=20$. This class of probabilistic constraints is called a chance constraint.

To do this, we first introduce a new callback and solve the system using the previous optimal solution

```@example control
constraint_condition(u,t,integrator) = u[1] - constraint[1]
constraint_affect!(integrator) = integrator.u[3] < constraint[2] ? terminate!(integrator) : nothing
constraint_cb = ContinuousCallback(constraint_condition, constraint_affect!, save_positions=(true,false));
constraint_cbs = CallbackSet(ground_cb, stop_cb, constraint_cb)

ensemblesol = solve(ensembleprob,Tsit5(),EnsembleThreads(), trajectories=500, callback=constraint_cbs)

begin
    plot(ensemblesol, vars = (1,3), lw=1,alpha=0.1)
    plot!(solve(remake(prob, u0=make_u0(minx)),Tsit5(), callback=constraint_cbs),
            vars=(1,3),label = nothing, c=:black, lw=3, ls=:dash)

    xlabel!("x [m]")
    ylabel!("y [m]")
    plot!(rectangle(27.5, 25, 5, 50), c=:red, label = nothing)
    plot!([constraint[1], constraint[1]],[0.0,constraint[2]], lw=5, c=:black)
    scatter!([25],[25],marker=:star, ms=10, label = nothing,c=:green)
    ylims!(0.0,50.0)
    xlims!(minx[1], 27.5)
end
```

That doesn't look good!

We now need a second observable for the system. In order to compute a probability of impact, we use an indicator function for if a trajectory impacts the wall. In other words, this functions returns 1 if the trajectory hits the wall and 0 otherwise.

```@example control
constraint_obs(sol,p) = sol[1,end] ≈ constraint[1] ? one(sol[1,end]) : zero(sol[1,end])
```

Using the previously computed optimal initial conditions, lets compute the probability of hitting this wall

```@example control
sm = SystemMap(remake(prob, u0=make_u0(minx)), Tsit5(),callback=constraint_cbs)
exprob = ExpectationProblem(sm, constraint_obs, h, gd; nout=1)
sol = solve(exprob, Koopman(),ireltol = 1e-5)
sol.u
```

We then setup the constraint function for NLopt just as before.

```@example control
function 𝔼_constraint(θ)
    prob = ODEProblem(ball!,make_u0(θ),tspan,p)
    sm = SystemMap(prob, Tsit5(),callback=constraint_cbs)
    exprob = ExpectationProblem(sm, constraint_obs, h, gd; nout=1)
    sol = solve(exprob, Koopman(),ireltol = 1e-5)
    sol.u
end
function 𝔼_constraint_nlopt(x,∇)
    length(∇) > 0 ? ForwardDiff.gradient!(∇, 𝔼_constraint,x) : nothing
    𝔼_constraint(x) - 0.01
end
```

Note that NLopt requires the constraint function to be of the form $g(x) \leq 0$. Hence, why we return `𝔼_constraint(x) - 0.01` for the 1% chance constraint.

The rest of the NLopt setup looks the same as before with the exception of adding the inequality constraint

```@example control
opt = Opt(:LD_MMA, 3)
opt.lower_bounds = [-100.0, 1.0, 10.0]
opt.upper_bounds = [0.0, 3.0, 50.0]
opt.xtol_rel = 1e-3
opt.min_objective = 𝔼_loss_nlopt
inequality_constraint!(opt,𝔼_constraint_nlopt, 1e-5)
(minf2,minx2,ret2) = NLopt.optimize(opt, [-1.0, 2.0, 50.0])
```

The probability of impacting the wall is now

```@example control
λ = 𝔼_constraint(minx2)
```
We can check if this is within tolerance by
```@example control
λ - 0.01 <= 1e-5
```

Again, we plot some Monte Carlo simulations from this result as follows

```@example control
ensembleprob = EnsembleProblem(remake(prob,u0 = make_u0(minx2)),prob_func=prob_func)
ensemblesol = solve(ensembleprob,Tsit5(),EnsembleThreads(),
                    trajectories=500, callback=constraint_cbs)

begin
    plot(ensemblesol, vars = (1,3), lw=1,alpha=0.1)
    plot!(solve(remake(prob, u0=make_u0(minx2)),Tsit5(), callback=constraint_cbs),
            vars=(1,3),label = nothing, c=:black, lw=3, ls=:dash)
    plot!([constraint[1], constraint[1]],[0.0,constraint[2]], lw=5, c=:black)

    xlabel!("x [m]")
    ylabel!("y [m]")
    plot!(rectangle(27.5, 25, 5, 50), c=:red, label = nothing)
    scatter!([25],[25],marker=:star, ms=10, label = nothing,c=:green)
    ylims!(0.0,50.0)
    xlims!(minx[1], 27.5)
end
```
