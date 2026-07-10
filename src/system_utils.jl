abstract type AbstractSystemMap end
"""
    SystemMap(prob; kwargs...)
    SystemMap(prob, alg; kwargs...)
    SystemMap(prob, alg, ensemblealg; kwargs...)

Represent the deterministic solution map `S(u0, p)` for a SciML problem.

Calling a `SystemMap` remakes `prob` with the supplied initial condition and
parameters, then solves the remade problem.

## Arguments

  - `prob`: SciML problem used as the template for repeated solves.
  - `alg`: Solver algorithm. If omitted, `solve` is called without an explicit
    algorithm.
  - `ensemblealg`: Ensemble algorithm used by ensemble-based expectation solves.
    Defaults to `EnsembleThreads()`.
  - `kwargs...`: Keyword arguments forwarded to `solve`.

## Fields

  - `prob`: Stored SciML problem.
  - `alg`: Stored solver algorithm or `nothing`.
  - `ensemblealg`: Stored ensemble algorithm.
  - `kwargs`: Stored solver keyword arguments.

## Returns

A callable `SystemMap`.
"""
struct SystemMap{
        DT <: SciMLBase.AbstractSciMLProblem,
        A <: Union{SciMLBase.AbstractODEAlgorithm, Nothing},
        EA <: SciMLBase.EnsembleAlgorithm, K,
    }
    prob::DT
    alg::A
    ensemblealg::EA
    kwargs::K
end
function SystemMap(prob; kwargs...)
    return SystemMap(prob, nothing, EnsembleThreads(), kwargs)
end
function SystemMap(prob, alg::SciMLBase.AbstractODEAlgorithm; kwargs...)
    return SystemMap(prob, alg, EnsembleThreads(), kwargs)
end
function SystemMap(
        prob, alg::SciMLBase.AbstractODEAlgorithm,
        ensemblealg::SciMLBase.EnsembleAlgorithm; kwargs...
    )
    return SystemMap(prob, alg, ensemblealg, kwargs)
end

function (sm::SystemMap{DT})(u0, p) where {DT}
    prob::DT = remake(
        sm.prob,
        u0 = convert(typeof(sm.prob.u0), u0),
        p = convert(typeof(sm.prob.p), p)
    )
    return solve(prob, sm.alg; sm.kwargs...)
end

"""
    ProcessNoiseSystemMap(prob, n, args...; kwargs...)

Represent a solution map for an SDE whose process noise is parameterized by
uncertain expansion coefficients.

Calling a `ProcessNoiseSystemMap` remakes `prob` with a
Kosambi-Karhunen-Loeve process-noise representation determined by the sampled
coefficients and then solves the remade problem.

## Arguments

  - `prob`: SciML problem used as the template for repeated solves.
  - `n`: Number of expansion terms in the process-noise representation.
  - `args...`: Positional arguments forwarded to `solve`.
  - `kwargs...`: Keyword arguments forwarded to `solve`.

## Fields

  - `prob`: Stored SciML problem.
  - `n`: Stored number of expansion terms.
  - `args`: Stored solver positional arguments.
  - `kwargs`: Stored solver keyword arguments.

## Returns

A callable `ProcessNoiseSystemMap`.
"""
struct ProcessNoiseSystemMap{DT <: SciMLBase.AbstractSciMLProblem, A, K} <: AbstractSystemMap
    prob::DT
    n::Int
    args::A
    kwargs::K
end
function ProcessNoiseSystemMap(prob, n, args...; kwargs...)
    return ProcessNoiseSystemMap(prob, n, args, kwargs)
end

function (sm::ProcessNoiseSystemMap{DT})(Z, p) where {DT}
    t0 = sm.prob.tspan[1]
    tend = sm.prob.tspan[2]
    function W(t)
        return sqrt(2) * (tend - t0) *
            sum(
            Z[k] * sin((k - 0.5) * pi * (t - t0) / (tend - t0)) / ((k - 0.5) * pi)
                for k in 1:length(Z)
        )
    end
    prob::DT = remake(
        sm.prob, p = convert(typeof(sm.prob.p), p),
        noise = DiffEqNoiseProcess.NoiseFunction{false}(t0, W)
    )
    return solve(prob, sm.args...; sm.kwargs...)
end
