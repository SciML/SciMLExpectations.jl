#Callable wrapper for DE solves. Enables seperation of args/kwargs...

abstract type AbstractSystemMap end

## Abstract System Map Interface

# Builds integrand for DEProblems
function build_integrand(prob::ExpectationProblem{F}, ::Koopman,
                         ::Val{false}) where {F <: AbstractSystemMap}
    @unpack S, g, h, d = prob
    function (x, p)
        ū, p̄ = h(x, p.x[1], p.x[2])
        g(S(ū, p̄), p̄) * pdf(d, x)
    end
end

function build_integrand(prob::ExpectationProblem{F}, ::Koopman,
                         ::Val{true}) where {F <: AbstractSystemMap}
    @unpack S, g, h, d = prob

    if prob.nout == 1 # TODO fix upstream in quadrature, expected sizes depend on quadrature method is requires different copying based on nout > 1
        set_result! = @inline function (dx, sol)
            dx[:] .= sol[:]
        end
    else
        set_result! = @inline function (dx, sol)
            dx .= reshape(sol[:, :], size(dx))
        end
    end

    prob_func = function (prob, i, repeat, x)  # TODO is it better to make prob/output funcs outside of integrand, then call w/ closure?
        u0, p = h((_make_view(x, i)), prob.u0, prob.p)
        remake(prob, u0 = u0, p = p)
    end

    output_func(sol, i, x) = (g(sol, sol.prob.p) * pdf(d, (_make_view(x, i))), false)

    function (dx, x, p) where {T}
        trajectories = size(x, 2)
        # TODO How to inject ensemble method in solve? currently in SystemMap, but does that make sense?
        ensprob = EnsembleProblem(S.prob; output_func = (sol, i) -> output_func(sol, i, x),
                                  prob_func = (prob, i, repeat) -> prob_func(prob, i,
                                                                             repeat, x))
        sol = solve(ensprob, S.args...; trajectories = trajectories, S.kwargs...)
        set_result!(dx, sol)
        nothing
    end
end

# solve expectation over DEProblem via MonteCarlo
function DiffEqBase.solve(exprob::ExpectationProblem{F},
                          expalg::MonteCarlo) where {F <: AbstractSystemMap}
    d = distribution(exprob)
    cov = input_cov(exprob)
    S = mapping(exprob)
    g = observable(exprob)

    prob_func = function (prob, i, repeat)
        u0, p = cov(rand(d), prob.u0, prob.p)
        remake(prob, u0 = u0, p = p)
    end

    output_func(sol, i) = (g(sol, sol.prob.p), false)

    monte_prob = EnsembleProblem(S.prob;
                                 output_func = output_func,
                                 prob_func = prob_func)
    sol = solve(monte_prob, S.args...; trajectories = expalg.trajectories, S.kwargs...)
    ExpectationSolution(mean(sol.u), nothing, nothing)
end

### Concrete System Maps


"""
```julia
SystemMap(prob, args...; kwargs...)
```

Representation of a system solution map for a given `prob::DEProblem`. `args` and `kwargs`
are forwarded to the equation solver.
"""
struct SystemMap{DT <: DiffEqBase.DEProblem,
                 A <: Union{SciMLBase.AbstractODEAlgorithm, Nothing},
                 EA <: SciMLBase.EnsembleAlgorithm, K}
    prob::DT
    alg::A
    ensemblealg::EA
    kwargs::K
end
function SystemMap(prob; kwargs...)
    SystemMap(prob, nothing, EnsembleThreads(), kwargs)
end
function SystemMap(prob, alg::SciMLBase.AbstractODEAlgorithm; kwargs...)
    SystemMap(prob, alg, EnsembleThreads(), kwargs)
end
function SystemMap(prob, alg::SciMLBase.AbstractODEAlgorithm,
                   ensemblealg::SciMLBase.EnsembleAlgorithm; kwargs...)
    SystemMap(prob, alg, ensemblealg, kwargs)
end

function (sm::SystemMap{DT})(u0, p) where {DT}
    prob::DT = remake(sm.prob,
                      u0 = convert(typeof(sm.prob.u0), u0),
                      p = convert(typeof(sm.prob.p), p))
    solve(prob, sm.alg; sm.kwargs...)
end

function ExpectationProblem(sm::SystemMap, g, h, d; nout = 1)
    ExpectationProblem(sm, g, h, d,
                       ArrayPartition(deepcopy(sm.prob.u0), deepcopy(sm.prob.p)),
                       nout)
end

"""
```julia
ProcessNoiseSystemMap(prob, args...; kwargs...)
```

Representation of a system solution map for a given `prob::DEProblem`. `args` and `kwargs`
are forwarded to the equation solver.
"""
struct ProcessNoiseSystemMap{DT <: DiffEqBase.DEProblem, A, K} <: AbstractSystemMap
    prob::DT
    n::Int
    args::A
    kwargs::K
end
ProcessNoiseSystemMap(prob, n, args...; kwargs...) = ProcessNoiseSystemMap(prob, n, args, kwargs)

function (sm::ProcessNoiseSystemMap{DT})(Z, p) where {DT}
    W(t) = sqrt(2) * sum(Z[k] * sin((k - 0.5) * pi * t) / ((k - 0.5) * pi) for k in 1:length(Z))
    prob::DT = remake(sm.prob, p = convert(typeof(sm.prob.p), p),
                               noise = NoiseFunction{false}(prob.tspan[1],W))
    solve(prob, sm.args...; sm.kwargs...)
end

function ExpectationProblem(sm::ProcessNoiseSystemMap, g, h; nout = 1)
    d = GenericDistribution((Normal() for i in 1:sm.n)...)
    ExpectationProblem(sm, g, h, d, deepcopy(sm.prob.p), nout)
end

function DiffEqBase.solve(exprob::ExpectationProblem{F},
                          expalg::MonteCarlo) where {F <: ProcessNoiseSystemMap}
    d = distribution(exprob)
    cov = input_cov(exprob)
    S = mapping(exprob)
    g = observable(exprob)

    prob_func = function (prob, i, repeat)
        Z, p = cov(rand(d), prob.u0, prob.p)
        function W(u,p,t)
            sqrt(2) *
            sum(Z[k] * sin((k - 0.5) * pi * t) / ((k - 0.5) * pi) for k in 1:length(Z))
        end
        remake(prob, p = p, noise = DiffEqNoiseProcess.NoiseFunction{false}(prob.tspan[1], W))
    end

    output_func(sol, i) = (g(sol, sol.prob.p), false)

    monte_prob = EnsembleProblem(S.prob;
                                 output_func = output_func,
                                 prob_func = prob_func)
    sol = solve(monte_prob, S.args...; trajectories = expalg.trajectories, S.kwargs...)
    ExpectationSolution(mean(sol.u), nothing, nothing)
end

function build_integrand(prob::ExpectationProblem{F}, ::Koopman,
                         ::Val{false}) where {F <: ProcessNoiseSystemMap}
    @unpack S, g, h, d = prob

    if prob.nout == 1 # TODO fix upstream in quadrature, expected sizes depend on quadrature method is requires different copying based on nout > 1
        set_result! = @inline function (dx, sol)
            dx[:] .= sol[:]
        end
    else
        set_result! = @inline function (dx, sol)
            dx .= reshape(sol[:, :], size(dx))
        end
    end

    prob_func = function (prob, i, repeat, x)
        Z, p = h((_make_view(x, i)), prob.u0, prob.p)
        function W(u,p,t)
            sqrt(2) *
            sum(Z[k] * sin((k - 0.5) * pi * t) / ((k - 0.5) * pi) for k in 1:length(Z))
        end
        remake(prob, p = p, noise = DiffEqNoiseProcess.NoiseFunction{false}(prob.tspan[1], W))
    end

    output_func(sol, i, x) = (g(sol, sol.prob.p) * pdf(d, (_make_view(x, i))), false)

    function (x, p) where {T}
        trajectories = size(x, 2)
        # TODO How to inject ensemble method in solve? currently in SystemMap, but does that make sense?
        ensprob = EnsembleProblem(S.prob; output_func = (sol, i) -> output_func(sol, i, x),
                                  prob_func = (prob, i, repeat) -> prob_func(prob, i,
                                                                             repeat, x))
        sol = solve(ensprob, S.args...; trajectories = trajectories, S.kwargs...)
        # set_result!(dx, sol)
        # nothing
        sol[:]
    end
end
