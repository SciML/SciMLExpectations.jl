#Callable wrapper for DE solves. Enables seperation of args/kwargs...

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
