
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
    
    T = promote_type([eltype.(u0_dist)..., eltype.(p_dist)...]...)

    (nu, usizes, umask) = _dist_mask.(u0_dist) |> _dist_mask_reduce
    (np, psizes, pmask) = _dist_mask.(p_dist) |> _dist_mask_reduce
    dist_mask = Bool.(vcat(umask, pmask))

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
        # we need to use this to play nicely with Zygote...
        _u = let u_it=1
                map(1:length(u0_dist)) do idx
                    ii = usizes[idx] > 1 ? (u_it:(u_it+usizes[idx]-1)) : u_it
                    u_it += usizes[idx]
                    # u[ii]
                    view(u, ii)
                end
            end
        _p = let p_it=1 
                map(1:length(p_dist)) do idx
                    ii = psizes[idx] > 1 ? (p_it:(p_it+psizes[idx]-1)) : p_it
                    p_it += psizes[idx]
                    # p[ii]
                    view(p, ii)
                end
            end
        return prod(_pdf(a,b) for (a,b) in zip(u0_dist,_u)) * prod(_pdf(a,b) for (a,b) in zip(p_dist,_p))
    end

    # sample from (joint) distribution
    samp_func() = comp_func(vcat(_rand.(u0_dist)...), vcat(_rand.(p_dist)...))

    # compute the bounds
    lb = isnothing(lower_bounds) ? to_quad(comp_func(vcat(_minimum.(u0_dist)...), vcat(_minimum.(p_dist)...))...)[1] : lower_bounds
    ub = isnothing(upper_bounds) ? to_quad(comp_func(vcat(_maximum.(u0_dist)...), vcat(_maximum.(p_dist)...))...)[1] : upper_bounds

    # compute "static" quadrature parameters
    p_quad = to_quad(comp_func(vcat(mean.(u0_dist)...), vcat(mean.(p_dist)...))...)[2]

    return ExpectationProblem(T,nout,g,to_quad,to_phys,f0_func,samp_func,comp_func,lb,ub,p_quad,prob,kwargs)
end

# Builds problem from (array) of u0 distribution(s)
function ExpectationProblem(g::Function, u0_dist, prob::ODEProblem, nout=1; kwargs...)
    T = promote_type(eltype.(u0_dist)...)
    return ExpectationProblem(g,u0_dist,Vector{T}(),prob,nout=nout; kwargs...)
end
