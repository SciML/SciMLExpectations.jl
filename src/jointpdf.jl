struct JointPdf{F,N,NU,NP,T}
    f::F
    idxs::ArrayPartition{Int, Tuple{NTuple{NU,Int}, NTuple{NP,Int}}}
    lb::NTuple{N,T}
    ub::NTuple{N,T}
end

function JointPdf(f::F, u_idx, u_lb, u_ub, p_idx, p_lb, p_ub) where F
    idxs = ArrayPartition(u_idx, p_idx)
    lb = tuple(u_lb..., p_lb...)
    ub = tuple(u_ub..., p_ub...)

    JointPdf(f, idxs, lb, ub)
end

function JointPdf(u0_pair::UT,p_pair::PT) where {UT <:Union{AbstractArray{<:Pair,1}, Tuple{Vararg{<:Pair}}}, 
                                                PT <:Union{AbstractArray{<:Pair,1}, Tuple{Vararg{<:Pair}}}}
    idxs = ArrayPartition(first.(u0_pair), first.(p_pair)) 
    dists = (last.(u0_pair)..., last.(p_pair)...)
    lb = tuple(minimum.(last.(u0_pair))..., minimum.(last.(p_pair))...)
    ub = tuple(maximum.(last.(u0_pair))..., maximum.(last.(p_pair))...)

    f = function(x)
        w = prod(pdf(a, b) for (a, b) in zip(dists, x))
    end
    JointPdf(f, idxs, lb, ub)
end

function JointPdf(u0s::U, ps::P) where {U,P}
    _, u_pair = transform_interface(zero(length(u0s)), u0s)
    _, p_pair = transform_interface(zero(length(u0s)), ps)

    JointPdf(u_pair, p_pair)
end

(j::JointPdf)(x) = j.f(x) 

bounds(j::JointPdf) = j.lb, j.ub
indices(j::JointPdf) = j.idxs