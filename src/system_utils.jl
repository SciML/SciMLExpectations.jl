abstract type AbstractSystemMap end
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

"""
```julia
ProcessNoiseSystemMap(prob, n, args...; kwargs...)
```

Representation of a system solution map for a given `prob::SDEProblem`. `args` and `kwargs`
are forwarded to the equation solver. `n` is the number of terms in the
Kosambi–Karhunen–Loève representation of the process noise.
"""
struct ProcessNoiseSystemMap{DT <: DiffEqBase.DEProblem, A, K} <: AbstractSystemMap
    prob::DT
    n::Int
    args::A
    kwargs::K
end
function ProcessNoiseSystemMap(prob, n, args...; kwargs...)
    ProcessNoiseSystemMap(prob, n, args, kwargs)
end

function (sm::ProcessNoiseSystemMap{DT})(Z, p) where {DT}
    t0 = prob.tspan[1]
    tend = prob.tspan[2]
    function W(t)
        sqrt(2) * (tend - t0) *
        sum(Z[k] * sin((k - 0.5) * pi * (t - t0) / (tend - t0)) / ((k - 0.5) * pi)
        for k in 1:length(Z))
    end
    prob::DT = remake(sm.prob, p = convert(typeof(sm.prob.p), p),
        noise = NoiseFunction{false}(prob.tspan[1], W))
    solve(prob, sm.args...; sm.kwargs...)
end
