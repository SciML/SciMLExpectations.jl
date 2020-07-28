abstract type AbstractExpectationAlgorithm <: DiffEqBase.DEAlgorithm end
struct Koopman <:AbstractExpectationAlgorithm end
struct MonteCarlo <: AbstractExpectationAlgorithm end

# tuplejoin from https://discourse.julialang.org/t/efficient-tuple-concatenation/5398/8
@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)

_rand(x::T) where T <: Sampleable = rand(x)
_rand(x) = x

function __make_map(prob::ODEProblem, args...; kwargs...)
    (u,p) -> solve(remake(prob,u0=u,p=p), args...; kwargs...)
end

function expectation(g::Function, prob::ODEProblem, u0, p, expalg::Koopman, args...;
                        u0_CoV=(u,p)->u, p_CoV=(u,p)->p,
                        maxiters=1000000,
                        batch=0,
                        quadalg=HCubatureJL(),
                        ireltol=1e-2, iabstol=1e-2,
                        nout=1,kwargs...)

    # S = __make_map(prob, args...; kwargs...)

    # expectation(g, S, u0, p, expalg, args...; u0_CoV=u0_CoV, p_CoV=p_CoV,
    #             maxiters=maxiters, batch=batch,
    #             ireltol=ireltol, iabstol=iabstol,
    #             quadalg=quadalg,
    #             nout=nout,kwargs...)

    # construct extended state space
    n_states = length(u0)
    n_params = length(p)
    ext_state = [u0; p]
    # ext_state = (u0...,p...)

    # find indices corresponding to distributions, check if sampleable and has non-zero support.
    dist_mask = collect(isa.(ext_state, Sampleable) .& (minimum.(ext_state) .!= maximum.(ext_state)))
    val_mask = .!(dist_mask)

    # get distributions and indx in extended state space
    dists = ext_state[dist_mask]

    # create numerical state space values
    # ext_state_val = minimum.(ext_state)
    T0 = promote_type(eltype.(ext_state)...)
    ext_state_val = [T0(minimum(v)) for v ∈ ext_state] |> collect  #collect needed for zygote for somereason. Otherwise is a tuple

    state_view = @view ext_state_val[dist_mask]
    param_view = @view ext_state_val[val_mask]

    if batch <= 1
        S = __make_map(prob, args...; kwargs...)
        integrand = function (x, p)
            ## Hack to avoid mutating array replacing ext_state_val[ext_state_dist_bitmask] .= x
            x_it = 0
            p_it = 0
            T = promote_type(eltype(x),eltype(p))
            esv = map(1:length(ext_state_val)) do idx
                dist_mask[idx] ? T(x[x_it+=1]) : T(p[p_it+=1])
            end

            _u0 = @view(esv[1:n_states])
            _p = @view(esv[n_states+1:end])

            # Koopman
            w = prod(pdf(a, b) for (a, b) in zip(dists, x))
            Ug = g(S(u0_CoV(_u0,_p), p_CoV(_u0,_p)))

            return Ug*w
        end
    else
        integrand = function (dx, x, p)
            trajectories = size(x, 2)
            T = promote_type(eltype(x),eltype(p))

            prob_func = function (prob, i, repeat) 
                x_it = 0
                p_it = 0  
                esv = map(1:length(ext_state_val)) do idx
                    dist_mask[idx] ? T(x[x_it+=1,i]) : T(p[p_it+=1])
                end

                u0_view = @view(esv[1:n_states])
                p_view = @view(esv[n_states+1:end])
                remake(prob, u0=u0_CoV(u0_view,p_view),p=p_CoV(u0_view,p_view))
            end

            output_func = function (sol, i)
                w = prod(pdf(a, b) for (a, b) in zip(dists, x[:,i]))
                Ug = g(sol)
                return Ug*w,false #Ug * w, false
            end

            ensembleprob = EnsembleProblem(prob, prob_func=prob_func, output_func=output_func)
            sol = solve(ensembleprob, args...;trajectories=trajectories,kwargs...)
            dx .= hcat(sol.u...) # Why do I need to hcat??? 
            nothing
        end
    end

    # TODO fix params usage
    lb = minimum.(dists)
    ub = maximum.(dists)

    T = promote_type(eltype(lb),eltype(ub), eltype(ext_state_val))
    intprob = QuadratureProblem(integrand, lb, ub, T.(param_view), batch=batch, nout=nout)
    sol = solve(intprob, quadalg, reltol=ireltol, abstol=iabstol, maxiters=maxiters)
end

# function expectation(g::Function, prob::ODEProblem, u0_f::Function, p_f::Function, p_quad, expalg::Koopman, args...;
#                         u0_CoV=(u,p)->u, p_CoV=(u,p)->p,
#                         maxiters=1000000,
#                         batch=0,
#                         quadalg=HCubatureJL(),
#                         ireltol=1e-2, iabstol=1e-2,
#                         nout=1,kwargs...)

#     if batch == 0 #TODO or 1
#         S = __make_map(prob, args...; kwargs...)
#         expectation(g, S, u0_f, p_f, p_quad, expalg, args...;
#                     u0_CoV=u0_CoV, p_CoV=p_CoV,
#                     maxiters=maxiters, batch=batch,
#                     ireltol=ireltol, iabstol=iabstol,
#                     quadalg=quadalg,
#                     nout=nout,kwargs...)
#     else
#         u0 = u0_f(p_quad)
#         p = p_f(p_quad)
#         n_states = length(u0)
#         ext_state = [u0; p]
    
#         # find indices corresponding to distributions, check if sampleable and has non-zero support.
#         dist_mask = collect(isa.(ext_state, Sampleable) .& (minimum.(ext_state) .!= maximum.(ext_state)))
#         val_mask = .!(dist_mask)

#         integrand = function (dx, x, p_quad)
#             trajectories = size(x, 2)

#             # reconstruct 
#             u0 = u0_f(p_quad)
#             p = p_f(p_quad)
#             ext_state = [u0; p]

#             dists = @view ext_state[dist_mask]

#             ext_state_val = minimum.(ext_state)
#             val_view = @view ext_state_val[val_mask]

#             T = promote_type(eltype(x),eltype(p_quad))

#             prob_func = function (prob, i, repeat) 
#                 x_it = 0
#                 p_it = 0  
#                 esv = map(1:length(ext_state_val)) do idx
#                     dist_mask[idx] ? T(x[x_it+=1,i]) : T(val_view[p_it+=1])
#                 end

#                 u0_view = @view(esv[1:n_states])
#                 p_view = @view(esv[n_states+1:end])
#                 remake(prob, u0=u0_CoV(u0_view,p_view),p=p_CoV(u0_view,p_view))
#             end

#             output_func = function (sol, i)
#                 w = prod(pdf(a, b) for (a, b) in zip(dists, x[:,i]))
#                 Ug = g(sol)
#                 return Ug*w,false #Ug * w, false
#             end

#             ensembleprob = EnsembleProblem(prob, prob_func=prob_func, output_func=output_func)
#             sol = solve(ensembleprob, args...;trajectories=trajectories,kwargs...)
#             if batch == 1
#                 # @show sol.u#sol.u[1]
#                 # return sol.u[1]
#                 dx[1] = sol.u[1]
#             else

#                 # return sol.u
#                 # @show hcat(sol.u...) 
#                 # return hcat(sol.u...) 
#                 dx .= hcat(sol.u...) # Why do I need to hcat???
#             end
#             # return dx
#         end

#         dists = @view ext_state[dist_mask]
#         lb = minimum.(dists)
#         ub = maximum.(dists)
#         T = promote_type(eltype(p_quad),eltype(lb),eltype(ub))
#         intprob = QuadratureProblem(integrand, T.(lb), T.(ub), T.(p_quad), batch=batch, nout=nout)
#         sol = solve(intprob, quadalg, reltol=ireltol, abstol=iabstol, maxiters=maxiters)
#     end

# end

# function expectation(g::Function, S::Function, u0, p, expalg::Koopman, args...;
#                         u0_CoV=(u,p)->u, p_CoV=(u,p)->p,
#                         maxiters=1000000,
#                         batch=0,
#                         quadalg=HCubatureJL(),
#                         ireltol=1e-2, iabstol=1e-2,
#                         nout=1,kwargs...)

#      # construct extended state space
#     n_states = length(u0)
#     n_params = length(p)
#     ext_state = [u0; p]

#     # find indices corresponding to distributions, check if sampleable and has non-zero support.
#     dist_mask = collect(isa.(ext_state, Sampleable) .& (minimum.(ext_state) .!= maximum.(ext_state)))
#     val_mask = .!(dist_mask)

#     # get distributions and indx in extended state space
#     dists = ext_state[dist_mask]

#     # create numerical state space values
#     ext_state_val = minimum.(ext_state)
#     state_view = @view ext_state_val[dist_mask]
#     param_view = @view ext_state_val[val_mask]

#     if batch <= 1
#         integrand = function (x, p)
#             ## Hack to avoid mutating array replacing ext_state_val[ext_state_dist_bitmask] .= x
#             x_it = 0
#             p_it = 0
#             T = promote_type(eltype(x),eltype(p))
#             esv = map(1:length(ext_state_val)) do idx
#                 dist_mask[idx] ? T(x[x_it+=1]) : T(p[p_it+=1])
#             end

#             _u0 = @view(esv[1:n_states])
#             _p = @view(esv[n_states+1:end])

#             # set values for indices corresponding to random variables
#             # state_view .= x
#             # _u0 = @view(ext_state_val[1:n_states])
#             # _p = @view(ext_state_val[n_states+1:end])

#             # Koopman
#             w = prod(pdf(a, b) for (a, b) in zip(dists, x))
#             Ug = g(S(u0_CoV(_u0,_p), p_CoV(_u0,_p)))

#             return Ug*w
#         end
#     end

#     # TODO fix params usage
#     lb = minimum.(dists)
#     ub = maximum.(dists)
#     # T = promote_type(eltype(p),eltype(lb),eltype(ub))
#     # intprob = QuadratureProblem(integrand, T.(lb), T.(ub), T.(p), batch=batch, nout=nout)
#     T = promote_type(eltype(param_view),eltype(lb),eltype(ub))
#     intprob = QuadratureProblem(integrand, T.(lb), T.(ub), T.(param_view), batch=batch, nout=nout)
#     sol = solve(intprob, quadalg, reltol=ireltol, abstol=iabstol, maxiters=maxiters)

#     sol
# end

# function expectation(g::Function, S::Function, u0_f::Function, p_f::Function, p_quad, expalg::Koopman, args...;
#                         u0_CoV=(u,p)->u, p_CoV=(u,p)->p,
#                         maxiters=1000000,
#                         batch=0,
#                         quadalg=HCubatureJL(),
#                         ireltol=1e-2, iabstol=1e-2,
#                         nout=1,kwargs...)

#     # construct extended state space
#     u0 = u0_f(p_quad)
#     p = p_f(p_quad)
#     n_states = length(u0)
#     ext_state = [u0; p]

#     # find indices corresponding to distributions, check if sampleable and has non-zero support.
#     dist_mask = collect(isa.(ext_state, Sampleable) .& (minimum.(ext_state) .!= maximum.(ext_state)))
#     val_mask = .!(dist_mask)

#     integrand = function (x, p_quad)
#         # reconstruct 
#         u0 = u0_f(p_quad)
#         p = p_f(p_quad)
#         ext_state = [u0; p]

#         dists = @view ext_state[dist_mask]

#         ext_state_val = minimum.(ext_state)
#         val_view = @view ext_state_val[val_mask]

#         # ## Hack to avoid mutating array replacing ext_state_val[dist_mask] .= x
#         x_it = 0
#         p_it = 0
#         T = promote_type(eltype(x),eltype(p_quad))
#         esv = map(1:length(ext_state_val)) do idx
#             dist_mask[idx] ? T(x[x_it+=1]) : T(val_view[p_it+=1])
#         end

#         u0_view = @view(esv[1:n_states])
#         p_view = @view(esv[n_states+1:end])

#         # Koopman
#         w =prod(pdf(a, b) for (a, b) in zip(dists, x))
#         Ug = g(S(u0_CoV(u0_view,p_view), p_CoV(u0_view,p_view)))
#         return Ug*w
#     end
    
#     dists = @view ext_state[dist_mask]
#     lb = minimum.(dists)
#     ub = maximum.(dists)
#     T = promote_type(eltype(p_quad),eltype(lb),eltype(ub))
#     intprob = QuadratureProblem(integrand, T.(lb), T.(ub), T.(p_quad), batch=batch, nout=nout)
#     sol = solve(intprob, quadalg, reltol=ireltol, abstol=iabstol, maxiters=maxiters)

#     sol
# end

function expectation(g::Function, prob::ODEProblem, u0, p, expalg::MonteCarlo, args...;
        trajectories,
        u0_CoV=(u,p)->u, p_CoV=(u,p)->p,
        kwargs...)

    prob_func = function (prob, i, repeat)
        _u0 = _rand.(u0)
        _p = _rand.(p)
        remake(prob, u0=u0_CoV(_u0,_p), p=p_CoV(_u0,_p))
    end

    output_func = (sol, i) -> (g(sol), false)

    monte_prob = EnsembleProblem(prob;
                output_func=output_func,
                prob_func=prob_func)
    sol = solve(monte_prob, args...;trajectories=trajectories,kwargs...)
    mean(sol.u)# , sol
end

function expectation(g::Function, S::Function, u0, p, expalg::MonteCarlo, args...;
                        trajectories,
                        u0_CoV=(u,p)->u, p_CoV=(u,p)->p,
                        kwargs...)

    function mc_run(u0,p)
        _u0 = _rand.(u0)
        _p = _rand.(p)
        g(S(u0_CoV(_u0,_p), p_CoV(_u0,_p)))
    end

    tot = mc_run(u0,p)
    for ii in 2:trajectories
        tot += mc_run(u0,p)
    end
    tot/trajectories
end

"""
    centralmoment(n, g, args...; kwargs) -> [n by 1 Array]

Computes the n central moments of the function g using the Koopman expectation.
The function is a wrapper over expectation, arguments can be piped through with
args and kwargs.

Return: n-length array of the 1 to n central moments

Note: The first central moment is, by definition, always 0

TODO: - add support for vector-valued g functions, currently assumes scalar 
      return values.
      - add tests
"""
function centralmoment(n::Int, g::Function, args...; kwargs...) 
    if n < 2 return Float64[] end

    # Compute the expectations of g, g^2, ..., g^n
    sol = expectation(x -> [g(x)^i for i in 1:n], args...; nout = n, kwargs...)
    exp_set = sol.u
    mu_g = popfirst!(exp_set)

    # Combine according to binomial expansion
    const_term(n) = (-1)^(n-1) * (n-1) * mu_g^n
    binom_term(n, k, mu, exp_gi) = binomial(n, k) * (-mu)^(n - k) * exp_gi
    binom_sum = function (exp_vals)
        m = length(exp_vals) + 1
        sum([binom_term(m, k + 1, mu_g, v) for (k,v) in enumerate(exp_vals)]) + const_term(m)
    end

    return [zero(exp_set[1]), [binom_sum(exp_set[1:i]) for i in 1:length(exp_set)]...]
end
