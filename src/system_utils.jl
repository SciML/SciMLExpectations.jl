#Callable wrapper for DE solves. Enables seperation of args/kwargs...

"""
```julia
SystemMap(prob, args...; kwargs...)
```

Representation of a system solution map for a given `prob::DEProblem`. `args` and `kwargs`
are forwarded to the equation solver.
"""
struct SystemMap{DT <: DiffEqBase.DEProblem, A, K}
    prob::DT
    args::A
    kwargs::K
end
SystemMap(prob, args...; kwargs...) = SystemMap(prob, args, kwargs)

function (sm::SystemMap{DT})(u0, p) where {DT}
    prob::DT = remake(sm.prob,
                      u0 = convert(typeof(sm.prob.u0), u0),
                      p = convert(typeof(sm.prob.p), p))
    solve(prob, sm.args...; sm.kwargs...)
end
