struct SystemMap{DT<:DiffEqBase.DEProblem,A,K}
    prob::DT
    args::A
    kwargs::K
end
SystemMap(prob, args...; kwargs...) = SystemMap(prob, args, kwargs)

function (sm::SystemMap{DT})(u0,p) where DT
    prob::DT = remake(sm.prob, u0 = u0, p = p)::DT
    solve(prob, sm.args...; sm.kwargs...)
end

# function (sm::SystemMap{DT})(u0p) where DT
#     sm(u0p...)
# end