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

abstract type AbstractExpectationAlgorithm <: DiffEqBase.DEAlgorithm end
struct Koopman{TS} <:AbstractExpectationAlgorithm where TS<:AbstractExpectationADAlgorithm
    sensealg::TS
end
Koopman() = Koopman(NonfusedAD())
struct MonteCarlo <: AbstractExpectationAlgorithm 
    trajectories::Int
end


#TODO need common return type for koopman() and MonteCarlo() solves
# solve expectation problem of generic callable functions via MonteCarlo
function DiffEqBase.solve(exprob::ExpectationProblem, expalg::MonteCarlo) 
    params = parameters(exprob)
    dist = distribution(exprob)
    g = observable(exprob)
    mean(g(rand(dist), params) for _ ∈ 1:expalg.trajectories)
end

# solve expectation over DEProblem via MonteCarlo
function DiffEqBase.solve(exprob::ExpectationProblem{F}, expalg::MonteCarlo) where F<:SystemMap
    d = distribution(exprob)
    cov = input_cov(exprob)
    S = mapping(exprob)
    g = observable(exprob)

    prob_func = function (prob, i, repeat)
        u0, p = cov(rand(d), prob.u0, prob.p)
        remake(prob, u0=u0, p=p)
    end

    output_func(sol, i) = (g(sol,sol.prob.p), false)

    monte_prob = EnsembleProblem(S.prob;
                output_func=output_func,
                prob_func=prob_func)
    sol = solve(monte_prob, S.args...;trajectories=expalg.trajectories,S.kwargs...)
    mean(sol.u)
end

# Solve Koopman expectation
function DiffEqBase.solve(prob::ExpectationProblem, expalg::Koopman, args...; 
                        maxiters=1000000,
                        batch=0,
                        quadalg=HCubatureJL(),
                        ireltol=1e-2, iabstol=1e-2,
                        kwargs...) where {A<:AbstractExpectationADAlgorithm}

    integrand = build_integrand(prob, expalg)
    lb, ub = extrema(prob.d)

    sol = integrate(quadalg, expalg.sensealg, integrand, lb, ub, prob.params;
            reltol=ireltol, abstol=iabstol, maxiters=maxiters, 
            nout = prob.nout, batch = batch, 
            kwargs...)

    return sol
end

# Integrate function to test new Adjoints, will need to roll up to Quadrature.jl
function integrate(quadalg, adalg::AbstractExpectationADAlgorithm, f, lb::TB, ub::TB, p; 
                        nout = 1, batch = 0,
                        kwargs...) where {TB}
    #TODO check batch iip type stability w/ QuadratureProblem{XXXX}
    prob = QuadratureProblem{false}(f,lb,ub,p; nout = nout, batch = batch)
    solve(prob, quadalg; kwargs...)
end

# defines adjoint via ∫∂/∂p f(x,p) dx
Zygote.@adjoint function integrate(quadalg, adalg::NonfusedAD, f::F, lb::T, ub::T, params::P; 
                            nout = 1, batch = 0, norm = norm,  
                            kwargs...) where {F,T,P}

    primal = integrate(quadalg, adalg, f, lb, ub, params; 
        norm = norm, nout = nout, batch = batch, 
        kwargs...)

    function integrate_pullbacks(Δ)
        function dfdp(x,params)
            _,back = Zygote.pullback(p->f(x,p),params)
            back(Δ)[1]
        end
        ∂p = integrate(quadalg, adalg, dfdp, lb, ub, params; 
            norm = norm, nout = nout*length(params), batch = batch,
            kwargs...)
        # ∂lb = -f(lb,params)  #needs correct for dim > 1
        # ∂ub = f(ub,params)
        return nothing, nothing, nothing, nothing, nothing, ∂p
    end 
    primal, integrate_pullbacks
end

# defines adjoint via ∫[f(x,p; ∂/∂p f(x,p)] dx, ie it fuses the primal, post the primal calculation
# has flag to only compute quad norm with respect to only the primal in the pull-back. Gives same quadrature points as doing forwarddiff
Zygote.@adjoint function integrate(quadalg, adalg::PostfusedAD, f::F, lb::T, ub::T, params::P; 
                            nout = 1, batch = 0, norm = norm,
                            kwargs...) where {F,T,P}

    primal = integrate(quadalg, adalg, f, lb, ub, params; 
        norm = norm, nout = nout, batch = batch, 
        kwargs...)

    _norm = adalg.norm_partials ? norm : primalnorm(nout, norm)

    function integrate_pullbacks(Δ)
        function dfdp(x,params)
            y, back = Zygote.pullback(p->f(x,p),params)
            [y; back(Δ)[1]]   #TODO need to match proper arrray type? promote_type???
        end
        ∂p = integrate(quadalg, adalg, dfdp, lb, ub, params; 
            norm = _norm, nout = nout + nout*length(params), batch = batch, 
            kwargs...)
        return nothing, nothing, nothing, nothing, nothing, @view ∂p[(nout+1):end]
    end 
    primal, integrate_pullbacks
end

# Fuses primal and partials prior to pullback, I doubt this will stick around based on required system evals. 
Zygote.@adjoint function integrate(quadalg, adalg::PrefusedAD, f::F, lb::T, ub::T, params::P; 
                            nout = 1, batch = 0, norm = norm,
                            kwargs...) where {F,T,P}
    # from Seth Axen via Slack
    # Does not work w/ ArrayPartition unless with following hack
    # Base.similar(A::ArrayPartition, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = similar(Array(A), T, dims)
    # TODO add ArrayPartition similar fix upstream, see https://github.com/SciML/RecursiveArrayTools.jl/issues/135
    ∂f_∂params(x, params) = only(Zygote.jacobian(p -> f(x, p), params))
	f_augmented(x, params) = [f(x, params); ∂f_∂params(x, params)...] #TODO need to match proper arrray type? promote_type???
	_norm = adalg.norm_partials ? norm : primalnorm(nout, norm)

    res = integrate(quadalg, adalg, f_augmented, lb, ub, params; 
        norm = _norm, nout = nout + nout*length(params), batch = batch,
        kwargs...)
	primal = first(res)
    function integrate_pullback(Δy)
		∂params = Δy .* conj.(@view(res[(nout+1):end]))
        return nothing, nothing, nothing, nothing, nothing, ∂params
    end 
    primal, integrate_pullback
end

# define norm function based only on primal part of fused integrand
function primalnorm(nout, fnorm)
    x->fnorm(@view x[1:nout])
end