import Base: rand, maximum, minimum, extrema
import Distributions: pdf

abstract type AbstractUncertaintyProblem end

struct GenericDistribution{TF, TRF, N, T}
    pdf_func::TF
    rand_func::TRF
    lb::NTuple{N,T}
    ub::NTuple{N,T}
    # TODO idxs::ArrayPartition{Int, Tuple{NTuple{NU,Int}, NTuple{NP,Int}}}  needs to get embedded in cov in ExpectationProblem
end

pdf(d::GenericDistribution, x) = d.pdf_func(x)
minimum(d::GenericDistribution) = d.lb
maximum(d::GenericDistribution) = d.ub
extrema(d::GenericDistribution) = minimum(d), maximum(d)
rand(d::GenericDistribution) = d.rand_func()

struct SystemMap{DT<:DiffEqBase.DEProblem}
    de_prob::DT
    args
    kwargs
end
SystemMap(prob, args...; kwargs...) = SystemMap(prob, args, kwargs)

function (sm::SystemMap{DT})(u0,p) where DT
    prob::DT = remake(sm.de_prob, u0 = u0, p = p)::DT
    solve(prob, sm.args...; sm.kwargs...)
end

struct ExpectationProblem{TG, TS, TH, TF} <: AbstractUncertaintyProblem
    # ∫ g(S(h(x,p)))*f(x)dx
    g::TG  # observable, g: 𝕐 → ℝ
    S::TS  # mapping,    S: 𝕐 × ℚ → 𝕐
    h::TH  # cov,        h: 𝕏 × ℙ → 𝕐 × ℚ
    f::TF  # pdf,        f: 𝕏 → ℝ
end  

# function GenericDistribution(u0_pair::UT,p_pair::PT) where {UT <:Union{AbstractArray{<:Pair,1}, Tuple{Vararg{<:Pair}}}, 
#                                                 PT <:Union{AbstractArray{<:Pair,1}, Tuple{Vararg{<:Pair}}}}
#     idxs = ArrayPartition(first.(u0_pair), first.(p_pair)) 
#     dists = (last.(u0_pair)..., last.(p_pair)...)
#     lb = tuple(minimum.(last.(u0_pair))..., minimum.(last.(p_pair))...)
#     ub = tuple(maximum.(last.(u0_pair))..., maximum.(last.(p_pair))...)

#     f = function(x)
#         w = prod(pdf(a, b) for (a, b) in zip(dists, x))
#     end
#     GenericDistribution(f, idxs, lb, ub)

#end