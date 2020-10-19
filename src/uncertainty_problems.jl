
struct ExpectationProblem{T, Tg, Tq, Tp, Tf, Ts, Tc, Tb, Tr, To, Tk} <: AbstractUncertaintyProblem
    Tscalar::Type{T}
    nout::Int64
    g::Tg
    to_quad::Tq
    to_phys::Tp
    f0_func::Tf
    samp_func::Ts
    comp_func::Tc
    quad_lb::Tb
    quad_ub::Tb
    p_quad::Tr
    ode_prob::To
    kwargs::Tk
end

DEFAULT_COMP_FUNC(x,p) = (x,p)

# Builds problem from (arrays) of u0 and p distribution(s)
function ExpectationProblem(g::Function, u0_dist, p_dist, prob::ODEProblem, nout=1; 
    comp_func=DEFAULT_COMP_FUNC, lower_bounds=nothing, upper_bounds=nothing, kwargs...)
    
    _xdist = u0_dist isa AbstractArray ? u0_dist : [u0_dist]
    _pdist = p_dist isa AbstractArray ? p_dist : [p_dist] 

    T = promote_type(eltype.(mean.([_xdist...,_pdist...]))...)

    # build shuffle/unshuffle functions
    usizes = Vector{Int64}([length(u) for u in _xdist])
    psizes = Vector{Int64}([length(p) for p in _pdist])
    nu = sum(usizes)
    np = sum(psizes)
    mx = vcat([repeat([u isa Distribution],length(u)) for u in _xdist]...)
    mp = vcat([repeat([p isa Distribution],length(p)) for p in _pdist]...)
    dist_mask = [mx..., mp...]

    # map physical x-state, p-params to quadrature state and params
    to_quad = function(x,p)
        esv = [x;p]
        return (view(esv,dist_mask), view(esv, .!(dist_mask)))
    end

    # map quadrature x-state, p-params to physical state and params
    to_phys = function(x,p)
        x_it, p_it = 0, 0
        esv = map(1:length(dist_mask)) do idx
            dist_mask[idx] ? T(x[x_it+=1]) : T(p[p_it+=1])
        end
        return (view(esv,1:nu), view(esv,(nu+1):(nu+np)))
    end

    # evaluate the f0 (joint) distribution
    f0_func = function(u,p)
        fu = prod([_pdf(dist, u[idx]) for (idx,dist) in zip(accumulated_range(usizes), _xdist)])
        fp = prod([_pdf(dist, p[idx]) for (idx,dist) in zip(accumulated_range(psizes), _pdist)])
        return fu * fp
    end

    # sample from (joint) distribution
    samp_func() = comp_func(vcat(_rand.(_xdist)...), vcat(_rand.(_pdist)...))

    # compute the bounds
    lb = isnothing(lower_bounds) ? to_quad(comp_func(vcat(_minimum.(_xdist)...), vcat(_minimum.(_pdist)...))...)[1] : lower_bounds
    ub = isnothing(upper_bounds) ? to_quad(comp_func(vcat(_maximum.(_xdist)...), vcat(_maximum.(_pdist)...))...)[1] : upper_bounds

    # compute "static" quadrature parameters
    p_quad = to_quad(comp_func(vcat(mean.(_xdist)...), vcat(mean.(_pdist)...))...)[2]


    return ExpectationProblem(T,nout,g,to_quad,to_phys,f0_func,samp_func,comp_func,lb,ub,p_quad,prob,kwargs)
end

# Builds problem from (array) of u0 distribution(s)
function ExpectationProblem(g::Function, u0_dist, prob::ODEProblem, nout=1; kwargs...)
    return ExpectationProblem(g,u0_dist,[],prob,nout=nout; kwargs...)
end
