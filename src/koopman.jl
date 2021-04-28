abstract type AbstractExpectationAlgorithm <: DiffEqBase.DEAlgorithm end
struct Koopman <:AbstractExpectationAlgorithm end
struct MonteCarlo <: AbstractExpectationAlgorithm end

abstract type AbstractExpectationADAlgorithm  end
struct NonfusedAD <: AbstractExpectationADAlgorithm end
struct PrefusedAD <: AbstractExpectationADAlgorithm
    norm_partials::Bool
end
PrefusedAD() = PrefusedAD(true)
struct PostfusedAD <: AbstractExpectationADAlgorithm 
    norm_partials::Bool
end
PostfusedAD() = PostfusedAD(true)

# Zygote.@adjoint function Zygote.literal_getproperty(
#         x::T,::Val{:norm_partials}) where {T<:Union{PrefusedAD, PostfusedAD}}
#     x.norm_partials, Δ->(T(Δ),)
# end

# Zygote.@adjoint function Zygote.literal_getfield(
#     x::T,::Val{:norm_partials}) where {T<:Union{PrefusedAD, PostfusedAD}}
# x.norm_partials, Δ->(T(Δ),)
# end

# function barrier. Need disbatch for zygote w/o mutation using zygote.buffer() ??? May not be needed as this is non-mutating?
# double check correctness, especially with CA w/ additional states
function inject(x, p::ArrayPartition{T,Tuple{TX, TP}}, dists_idx) where {T, TX, TP}
    it::Int = 0      #type annotation to ensure boxed variable is type stable. See: https://docs.julialang.org/en/v1/manual/performance-tips/#man-performance-captured
    dist_it::Int = 1
    state = map(p.x[1]) do i
                it += 1
                if dist_it <= length(dists_idx.x[1]) && dists_idx.x[1][dist_it[1]] == it
                    dist_it += 1
                    return eltype(TX)(x[dist_it - 1])
                else
                    return eltype(TX)(p.x[1][it])
                end
            end

    it = 0
    dist_it = 1
    param = map(p.x[2]) do i
                it+=1
                if dist_it <= length(dists_idx.x[2]) && dists_idx.x[2][dist_it] == it
                    dist_it+=1
                    return eltype(TP)(x[dist_it+length(dists_idx.x[1]) - 1])
                else
                    return eltype(TP)(p.x[2][it])
                end
            end
    ArrayPartition(state, param)
end

function transform_interface(prob_x::TX, x) where TX
    dists = filter(y-> !isa(y[2],Number), (enumerate(x)...,))
    x_pair = map(y->Pair(y[1],y[2]), dists) 
    
    it::Int = 0
    _x = map(prob_x) do i
                it+=1
                if x[it] isa Number
                    return eltype(TX)(x[it])
                else
                    return zero(eltype(TX))
                end
            end
    
    _x, x_pair
end

function build_integrand(g::F, prob::deT, u0_pair, p_pair, args...; kwargs...) where {F, deT}
    dists = (last.(u0_pair)..., last.(p_pair)...)
    dists_idx = ArrayPartition(first.(u0_pair), first.(p_pair)) 

    integrand = function(x,p)
        p2 = inject(x, p, dists_idx)
        prob_update::deT = remake(prob, u0 = p2.x[1], p = p2.x[2])  #deT for compiler hint for stability
        Sx = solve(prob_update, args...; kwargs...)
        # push!(results, Sx)
        w = prod(pdf(a, b) for (a, b) in zip(dists, x))
        g(Sx)*w
    end
end


# g::Function, prob::DiffEqBase.AbstractODEProblem, u0, p, expalg::Koopman, args...;
#                         u0_CoV=(u,p)->u, p_CoV=(u,p)->p,
function expectation(g::F, prob::deT, u0, p, args...; 
                        kwargs...) where {F,deT}

    _u0, u0_pair = transform_interface(prob.u0, u0)
    _p, p_pair = transform_interface(prob.p, p)
    prob_update::deT = remake(prob, u0 = _u0, p = _p)

    return expectation(g, prob_update, u0_pair, p_pair, args...; kwargs...)
end

function expectation(g::F, prob::deT, u0_pair::uT, p_pair::pT, expalg::Koopman, args...; 
                        adalg::A = NonfusedAD(),
                        maxiters=1000000,
                        batch=0,
                        quadalg=HCubatureJL(),
                        ireltol=1e-2, iabstol=1e-2,
                        nout=1,
                        kwargs...) where {F,deT, 
                                          uT <:Union{AbstractArray{<:Pair,1}, Tuple{Vararg{<:Pair}}}, 
                                          pT <:Union{AbstractArray{<:Pair,1}, Tuple{Vararg{<:Pair}}},
                                          A<:AbstractExpectationADAlgorithm}

    # determine DE solve return type and construct array to store results
    # TODO integrate into build_integrand
    #solT = Core.Compiler.return_type(Core.kwfunc(solve), 
    #                                Tuple{typeof(values(kwargs)), typeof(solve),
    #                                typeof(prob), typeof.(args)...})
    # @show Core.Compiler.return_tupe(g, solT)
    # results = solT[]

    quad_p = ArrayPartition(deepcopy(prob.u0), deepcopy(prob.p))

    integrand = build_integrand(g, prob, u0_pair, p_pair, args...; kwargs...)
 
    # tuple bounds required for type stability w/ HCubature
    lb = tuple(minimum.(last.(u0_pair))..., minimum.(last.(p_pair))...)
    ub = tuple(maximum.(last.(u0_pair))..., maximum.(last.(p_pair))...)

    sol = myintegrate(quadalg, adalg, integrand, lb, ub, quad_p;
            nout = nout, batch = batch, reltol=ireltol, abstol=iabstol, maxiters=maxiters, kwargs...)

    return sol#, EnsembleSolution(results,0.0, true)
end

function myintegrate(quadalg, adalg::AbstractExpectationADAlgorithm, f::F, lb::T, ub::T, p::P; 
                        nout = 1, batch = 0,
                        kwargs...) where {F,T,P}
    #TODO check batch iip type stability
    # iip = batch > 1
    prob = QuadratureProblem{false}(f,lb,ub,p; nout = nout, batch = batch)
    res = solve(prob, quadalg; kwargs...)
    res.u #TODO revert to returning full solution, i.e. res
end

function primalnorm(nout, norm)
    x->norm(@view x[1:nout])
end

Zygote.@adjoint function myintegrate(quadalg, adalg::NonfusedAD, f::F, lb::T, ub::T, params::P; 
                            nout = 1, batch = 0, norm = norm,  
                            kwargs...) where {F,T,P}
    @show "nonfusedAD"
    primal = myintegrate(quadalg, adalg, f, lb, ub, params; 
        norm = norm, nout = nout, batch = batch, 
        kwargs...)
    @show "primal done"
    function myintegrate_pullbacks(Δ)
        function dfdp(x,params)
            _,back = Zygote.pullback(p->f(x,p),params)
            back(Δ)[1]
        end
        ∂p = myintegrate(quadalg, adalg, dfdp, lb, ub, params; 
            norm = norm, nout = nout*length(params), batch = batch,
            kwargs...)
        # ∂lb = -f(lb,params)  #needs correct for dim > 1
        # ∂ub = f(ub,params)
        return nothing, nothing, nothing, nothing, nothing, ∂p
    end 
    primal, myintegrate_pullbacks
end

Zygote.@adjoint function myintegrate(quadalg, adalg::PostfusedAD, f::F, lb::T, ub::T, params::P; 
                            nout = 1, batch = 0, norm = norm,
                            kwargs...) where {F,T,P}
	@show "post fusedAD"
    primal = myintegrate(quadalg, adalg, f, lb, ub, params; 
        norm = norm, nout = nout, batch = batch, 
        kwargs...)
    @show "primal done"

    _norm = adalg.norm_partials ? norm : primalnorm(nout, norm)

    function myintegrate_pullbacks(Δ)
        function dfdp(x,params)
            y, back = Zygote.pullback(p->f(x,p),params)
            [y; back(Δ)[1]]   #TODO need to match proper arrray type? promote_type???
        end
        ∂p = myintegrate(quadalg, adalg, dfdp, lb, ub, params; 
            norm = _norm, nout = nout + nout*length(params), batch = batch, 
            kwargs...)
        return nothing, nothing, nothing, nothing, nothing, @view ∂p[(nout+1):end]
    end 
    primal, myintegrate_pullbacks
end

# from Seth Axen via Slack
# Does not work w/ ArrayPartition unless with following hack
# TODO add ArrayPartition similar fix upstream, see https://github.com/SciML/RecursiveArrayTools.jl/issues/135
# Base.similar(A::ArrayPartition, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = similar(Array(A), T, dims)
Zygote.@adjoint function myintegrate(quadalg, adalg::PrefusedAD, f::F, lb::T, ub::T, params::P; 
                            nout = 1, batch = 0, norm = norm,
                            kwargs...) where {F,T,P}
	@show "pre fusedAD"
    ∂f_∂params(x, params) = only(Zygote.jacobian(p -> f(x, p), params))
	f_augmented(x, params) = [f(x, params); ∂f_∂params(x, params)...] #TODO need to match proper arrray type? promote_type???
	_norm = adalg.norm_partials ? norm : primalnorm(nout, norm)

    res = myintegrate(quadalg, adalg, f_augmented, lb, ub, params; 
        norm = _norm, nout = nout + nout*length(params), batch = batch,
        kwargs...)
	primal = first(res)
    function integrate_pullback(Δy)
		∂params = Δy .* conj.(@view(res[(nout+1):end]))
        return nothing, nothing, nothing, nothing, nothing, ∂params
    end 
    primal, integrate_pullback
end

aasdf(x) = x
# 

# # tuplejoin from https://discourse.julialang.org/t/efficient-tuple-concatenation/5398/8
# @inline tuplejoin(x) = x
# @inline tuplejoin(x, y) = (x..., y...)
# @inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)

# _rand(x::T) where T <: Sampleable = rand(x)
# _rand(x) = x

# function __make_map(prob::ODEProblem, args...; kwargs...)
#     (u,p) -> solve(remake(prob,u0=u,p=p), args...; kwargs...)
# end

# function expectation(g::Function, prob::DiffEqBase.AbstractODEProblem, u0, p, expalg::Koopman, args...;
#                         u0_CoV=(u,p)->u, p_CoV=(u,p)->p,
#                         maxiters=1000000,
#                         batch=0,
#                         quadalg=HCubatureJL(),
#                         ireltol=1e-2, iabstol=1e-2,
#                         nout=1,kwargs...)

#     # construct extended state space
#     n_states = length(u0)
#     n_params = length(p)
#     ext_state = [u0; p]
#     # ext_state = (u0...,p...)

#     # find indices corresponding to distributions, check if sampleable and has non-zero support.
#     dist_mask = collect(isa.(ext_state, Union{KernelDensity.AbstractKDE,
#                                               Distributions.Sampleable}) .&
#                                               (minimum.(ext_state) .!= maximum.(ext_state)))
#     val_mask = .!(dist_mask)

#     # get distributions and indx in extended state space
#     dists = ext_state[dist_mask]

#     # create numerical state space values
#     # ext_state_val = minimum.(ext_state)
#     T0 = promote_type(eltype.(ext_state)...)
#     ext_state_val = [T0(minimum(v)) for v ∈ ext_state] |> collect  #collect needed for zygote for somereason. Otherwise is a tuple

#     state_view = @view ext_state_val[dist_mask]
#     param_view = @view ext_state_val[val_mask]

#     if batch <= 1
#         S = __make_map(prob, args...; kwargs...)
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

#             # Koopman
#             w = prod(pdf(a, b) for (a, b) in zip(dists, x))
#             Ug = g(S(u0_CoV(_u0,_p), p_CoV(_u0,_p)))

#             return Ug*w
#         end
#     else
#         integrand = function (dx, x, p)
#             trajectories = size(x, 2)
#             T = promote_type(eltype(x),eltype(p))

#             prob_func = function (prob, i, repeat) 
#                 x_it = 0
#                 p_it = 0  
#                 esv = map(1:length(ext_state_val)) do idx
#                     dist_mask[idx] ? T(x[x_it+=1,i]) : T(p[p_it+=1])
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
#             dx .= hcat(sol.u...) # Why do I need to hcat??? 
#             nothing
#         end
#     end

#     # TODO fix params usage
#     lb = minimum.(dists)
#     ub = maximum.(dists)

#     T = promote_type(eltype(lb),eltype(ub), eltype(ext_state_val))
#     intprob = QuadratureProblem(integrand, lb, ub, T.(param_view), batch=batch, nout=nout)
#     sol = solve(intprob, quadalg, reltol=ireltol, abstol=iabstol, maxiters=maxiters)
# end

# function expectation(g::Function, prob::DiffEqBase.AbstractODEProblem, u0, p, expalg::MonteCarlo, args...;
#         trajectories,
#         u0_CoV=(u,p)->u, p_CoV=(u,p)->p,
#         kwargs...)

#     prob_func = function (prob, i, repeat)
#         _u0 = _rand.(u0)
#         _p = _rand.(p)
#         remake(prob, u0=u0_CoV(_u0,_p), p=p_CoV(_u0,_p))
#     end

#     output_func = (sol, i) -> (g(sol), false)

#     monte_prob = EnsembleProblem(prob;
#                 output_func=output_func,
#                 prob_func=prob_func)
#     sol = solve(monte_prob, args...;trajectories=trajectories,kwargs...)
#     mean(sol.u)# , sol
# end

# """
#     centralmoment(n, g, args...; kwargs) -> [n by 1 Array]

# Computes the n central moments of the function g using the Koopman expectation.
# The function is a wrapper over expectation, arguments can be piped through with
# args and kwargs.

# Return: n-length array of the 1 to n central moments

# Note: The first central moment is, by definition, always 0

# TODO: - add support for vector-valued g functions, currently assumes scalar 
#       return values.
#       - add tests
# """
# function centralmoment(n::Int, g::Function, args...; kwargs...) 
#     if n < 2 return Float64[] end

#     # Compute the expectations of g, g^2, ..., g^n
#     sol = expectation(x -> [g(x)^i for i in 1:n], args...; nout = n, kwargs...)
#     exp_set = sol.u
#     mu_g = popfirst!(exp_set)

#     # Combine according to binomial expansion
#     const_term(n) = (-1)^(n-1) * (n-1) * mu_g^n
#     binom_term(n, k, mu, exp_gi) = binomial(n, k) * (-mu)^(n - k) * exp_gi
#     binom_sum = function (exp_vals)
#         m = length(exp_vals) + 1
#         sum([binom_term(m, k + 1, mu_g, v) for (k,v) in enumerate(exp_vals)]) + const_term(m)
#     end

#     return [zero(exp_set[1]), [binom_sum(exp_set[1:i]) for i in 1:length(exp_set)]...]
# end
