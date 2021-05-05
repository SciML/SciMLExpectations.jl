import Base: rand, maximum, minimum, extrema
import Distributions: pdf

abstract type AbstractUncertaintyProblem end

struct GenericDistribution{TF, TRF, TLB, TUB}
    pdf_func::TF
    rand_func::TRF
    lb::TLB
    ub::TUB
end

# function GenericDistribution(dists...)
#     # TODO add support to mix univariate and MV distrributions???
#     pdf_func(x) = prod(pdf(f,y) for (f,y) in zip(dists,x))
#     rand_func() =  [rand(d) for d in dists] #mapreduce(rand, vcat, dists)
#     lb = minimum.(dists)
#     ub = maximum.(dists)
#     GenericDistribution(pdf_func, rand_func, lb, ub)
# end

pdf(d::GenericDistribution, x) = d.pdf_func(x)
minimum(d::GenericDistribution) = d.lb
maximum(d::GenericDistribution) = d.ub
extrema(d::GenericDistribution) = minimum(d), maximum(d)
rand(d::GenericDistribution) = d.rand_func()

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

# function (sm::SystemMap{DT})(u0,p) where DT
#     sm(u0, p)
# end

struct ExpectationProblem{TS, TG, TH, TF, TP} <: AbstractUncertaintyProblem
    # ∫ g(S(h(x,p)))*f(x)dx
    S::TS  # mapping,                 S: 𝕐 × ℚ → 𝕐
    g::TG  # observable(output_func), g: 𝕐 → ℝ
    h::TH  # cov(input_func),         h: 𝕏 × ℙ → 𝕐 × ℚ
    d::TF  # distribution,            pdf(d,x): 𝕏 → ℝ
    params::TP
end 

function ExpectationProblem(S, pdist, params)
    g(x) = x
    h(x,u,p) = x,p
    ExpectationProblem(S,g,h,pdist,ArrayPartition(eltype(params)[], params))
end

ExpectationProblem(sm::SystemMap, g, h, d) = 
    ExpectationProblem(sm,g,h,d,ArrayPartition(deepcopy(sm.prob.u0),deepcopy(sm.prob.p)))


function build_integrand(ep::ExpectationProblem)
    @unpack S, g, h, d = ep
    Ug = g∘S∘h
    function(x,p)
        g(S(h(x, p.x[1], p.x[2])...))*pdf(d,x)
    end
end


# function ExpectationProblem(g, S::SystemMap, u0_pair::uT, p_pair::pT) 
#                                 where {uT <:Union{AbstractArray{<:Pair,1}, Tuple{Vararg{<:Pair}}}, 
#                                        pT <:Union{AbstractArray{<:Pair,1}, Tuple{Vararg{<:Pair}}}}

#     idxs = ArrayPartition(first.(u0_pair), first.(p_pair)) 
# end





# integrand(ep::ExpectationProblem) = ep.g(ep.S(ep.h(x,p)...))*ep.f(x)
# function ExpectationProblem(g::TG, S::TS, 

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