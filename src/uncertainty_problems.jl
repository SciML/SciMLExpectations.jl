
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

# make a problem out of 
function ExpectationProblem(g::Function, u0_dist, p_dist, prob::ODEProblem, nout=1; 
    comp_func=DEFAULT_COMP_FUNC, kwargs...)

    T = promote_type(eltype.(mean.([u0_dist...,p_dist...]))...)

    # build shuffle/unshuffle functions
    usizes = [length(u) for u in u0_dist]
    psizes = [length(p) for p in p_dist]
    nu = sum(usizes)
    np = sum(psizes)
    mx = vcat([repeat([u isa Distribution],length(u)) for u in u0_dist]...)
    mp = vcat([repeat([p isa Distribution],length(p)) for p in p_dist]...)
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
        fu = prod([_pdf(dist, u[idx]) for (idx,dist) in zip(accumulated_range(usizes), u0_dist)])
        fp = prod([_pdf(dist, p[idx]) for (idx,dist) in zip(accumulated_range(psizes), p_dist)])
        return fu * fp
    end

    # sample from (joint) distribution
    samp_func() = comp_func(vcat(_rand.(u0_dist)...), vcat(_rand.(p_dist)...))

    # compute the bounds
    lb = to_quad(comp_func(minimum.(u0_dist), minimum.(p_dist))...)[1]
    ub = to_quad(comp_func(maximum.(u0_dist), maximum.(p_dist))...)[1]

    # compute "static" quadrature parameters
    p_quad = to_quad(comp_func(mean.(u0_dist), mean.(p_dist))...)[2]


    return ExpectationProblem(Tscalar,nout,g,to_quad,to_phys,f0_func,samp_func,comp_func,lb,ub,p_quad,prob,kwargs)
end

