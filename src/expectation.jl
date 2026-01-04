abstract type AbstractExpectationADAlgorithm end
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

"""
```julia
Koopman()
```
"""
struct Koopman{TS} <:
    AbstractExpectationAlgorithm where {TS <: AbstractExpectationADAlgorithm}
    sensealg::TS
end
Koopman() = Koopman(NonfusedAD())

"""
```julia
MonteCarlo(trajectories::Int)
```
"""
struct MonteCarlo <: AbstractExpectationAlgorithm
    trajectories::Int
end

# Builds integrand for arbitrary functions
function build_integrand(prob::ExpectationProblem, ::Koopman, mid, p, ::Nothing)
    @unpack g, d = prob
    function integrand_koopman(x, p)
        return g(x, p) * pdf(d, x)
    end
    return IntegralFunction{false}(integrand_koopman, nothing)
end

# Builds integrand for DEProblems
function build_integrand(
        prob::ExpectationProblem{F}, ::Koopman, mid, p,
        ::Nothing
    ) where {F <: SystemMap}
    @unpack S, g, h, d = prob
    function integrand_koopman_systemmap(x, p)
        ū, p̄ = h(x, p.x[1], p.x[2])
        return g(S(ū, p̄), p̄) * pdf(d, x)
    end
    return IntegralFunction{false}(integrand_koopman_systemmap, nothing)
end

function _make_view(x::Union{Vector{T}, Adjoint{T, Vector{T}}}, i) where {T}
    return @view x[i]
end

function _make_view(x, i)
    return @view x[:, i]
end

function build_integrand(
        prob::ExpectationProblem{F}, ::Koopman, mid, p,
        batch::Integer
    ) where {F <: SystemMap}
    @unpack S, g, h, d = prob

    function prob_func(prob, i, repeat, x)  # TODO is it better to make prob/output funcs outside of integrand, then call w/ closure?
        u0, _p = h((_make_view(x, i)), prob.u0, prob.p)
        return remake(prob, u0 = u0, p = _p)
    end

    output_func(sol, i, x) = (g(sol, sol.prob.p) * pdf(d, (_make_view(x, i))), false)

    function integrand_koopman_systemmap_batch(x, _)
        trajectories = size(x)[end]
        # TODO How to inject ensemble method in solve? currently in SystemMap, but does that make sense?
        ensprob = EnsembleProblem(
            S.prob; output_func = (sol, i) -> output_func(sol, i, x),
            prob_func = (prob, i, repeat) -> prob_func(
                prob, i,
                repeat, x
            )
        )
        return solve(ensprob, S.alg, S.ensemblealg; trajectories = trajectories, S.kwargs...)
    end
    function integrand_koopman_systemmap_batch!(dx, x, p)
        dx .= integrand_koopman_systemmap_batch(x, p)
        return nothing
    end
    proto_sol = integrand_koopman_systemmap_batch(reshape(collect(mid), size(mid)..., 1), p)
    prototype = Array(proto_sol)
    return BatchIntegralFunction{true}(integrand_koopman_systemmap_batch!, prototype; max_batch = batch)
end

function build_integrand(
        prob::ExpectationProblem{F}, ::Koopman, mid,
        ::Nothing
    ) where {F <: ProcessNoiseSystemMap}
    error("Batching is required for Koopman ProcessNoiseSystemMap")
end

function build_integrand(
        prob::ExpectationProblem{F}, ::Koopman, mid, p,
        batch::Integer
    ) where {F <: ProcessNoiseSystemMap}
    @unpack S, g, h, d = prob

    prob_func = function (prob, i, repeat, x)
        Z, p = h((_make_view(x, i)), prob.u0, prob.p)
        function W(u, p, t)
            return sqrt(2) *
                sum(Z[k] * sin((k - 0.5) * pi * t) / ((k - 0.5) * pi) for k in 1:length(Z))
        end
        return remake(
            prob, p = p,
            noise = DiffEqNoiseProcess.NoiseFunction{false}(prob.tspan[1], W)
        )
    end

    output_func(sol, i, x) = (g(sol, sol.prob.p) * pdf(d, (_make_view(x, i))), false)

    function integrand_koopman_processnoisesystemmap_batch(x, p)
        trajectories = size(x)[end]
        # TODO How to inject ensemble method in solve? currently in SystemMap, but does that make sense?
        ensprob = EnsembleProblem(
            S.prob; output_func = (sol, i) -> output_func(sol, i, x),
            prob_func = (prob, i, repeat) -> prob_func(
                prob, i,
                repeat, x
            )
        )
        return solve(ensprob, S.args...; trajectories = trajectories, S.kwargs...)
    end
    function integrand_koopman_processnoisesystemmap_batch!(dx, x, p)
        dx .= integrand_koopman_processnoisesystemmap_batch(x, p)
        return nothing
    end
    proto_sol = integrand_koopman_processnoisesystemmap_batch(reshape(collect(mid), size(mid)..., 1), p)
    prototype = Array(proto_sol)
    return BatchIntegralFunction{true}(
        integrand_koopman_processnoisesystemmap_batch!, prototype; max_batch = batch
    )
end

"""
```julia
solve(exprob::ExpectationProblem, expalg::MonteCarlo)
```

Solve an `ExpectationProblem` using Monte Carlo integration.
"""
function DiffEqBase.solve(exprob::ExpectationProblem, expalg::MonteCarlo)
    params = parameters(exprob)
    dist = distribution(exprob)
    g = observable(exprob)
    return ExpectationSolution(
        mean(g(rand(dist), params) for _ in 1:(expalg.trajectories)),
        nothing, nothing
    )
end

function DiffEqBase.solve(
        exprob::ExpectationProblem{F},
        expalg::MonteCarlo
    ) where {F <: SystemMap}
    d = distribution(exprob)
    cov = input_cov(exprob)
    S = mapping(exprob)
    g = observable(exprob)

    prob_func = function (prob, i, repeat)
        u0, p = cov(rand(d), prob.u0, prob.p)
        return remake(prob, u0 = u0, p = p)
    end

    output_func(sol, i) = (g(sol, sol.prob.p), false)

    monte_prob = EnsembleProblem(
        S.prob;
        output_func = output_func,
        prob_func = prob_func
    )
    sol = solve(
        monte_prob, S.alg, S.ensemblealg; trajectories = expalg.trajectories,
        S.kwargs...
    )
    return ExpectationSolution(mean(sol.u), nothing, nothing)
end
function DiffEqBase.solve(
        exprob::ExpectationProblem{F},
        expalg::MonteCarlo
    ) where {F <: ProcessNoiseSystemMap}
    d = distribution(exprob)
    cov = input_cov(exprob)
    S = mapping(exprob)
    g = observable(exprob)

    prob_func = function (prob, i, repeat)
        Z, p = cov(rand(d), prob.u0, prob.p)
        function W(u, p, t)
            return sqrt(2) *
                sum(Z[k] * sin((k - 0.5) * pi * t) / ((k - 0.5) * pi) for k in 1:length(Z))
        end
        return remake(
            prob, p = p,
            noise = DiffEqNoiseProcess.NoiseFunction{false}(prob.tspan[1], W)
        )
    end

    output_func(sol, i) = (g(sol, sol.prob.p), false)

    monte_prob = EnsembleProblem(
        S.prob;
        output_func = output_func,
        prob_func = prob_func
    )
    sol = solve(monte_prob, S.args...; trajectories = expalg.trajectories, S.kwargs...)
    return ExpectationSolution(mean(sol.u), nothing, nothing)
end

"""
```julia
solve(exprob::ExpectationProblem, expalg::Koopman;
      maxiters = 1000000, batch = nothing, quadalg = HCubatureJL(),
      ireltol = 1e-2, iabstol = 1e-2, kwargs...)
```

Solve an `ExpectationProblem` using Koopman integration.
"""
function DiffEqBase.solve(
        prob::ExpectationProblem, expalg::Koopman, args...;
        maxiters = 1000000,
        batch = nothing,
        quadalg = HCubatureJL(),
        ireltol = 1.0e-2, iabstol = 1.0e-2,
        kwargs...
    )
    domain = extrema(prob.d)
    integrand = build_integrand(prob, expalg, sum(domain) / 2, prob.params, batch)

    sol = integrate(
        quadalg, expalg.sensealg, integrand, domain, prob.params;
        reltol = ireltol, abstol = iabstol, maxiters = maxiters,
        kwargs...
    )

    return ExpectationSolution(sol.u, sol.resid, sol)
end

# Integrate function to test new Adjoints, will need to roll up to Integrals.jl
function integrate(
        quadalg, adalg::AbstractExpectationADAlgorithm, f, domain, p;
        kwargs...
    )
    prob = IntegralProblem(f, domain, p)
    return solve(prob, quadalg; kwargs...)
end

# defines adjoint via ∫∂/∂p f(x,p) dx
Zygote.@adjoint function integrate(
        quadalg, adalg::NonfusedAD, f::F, domain,
        params::P;
        #    norm = norm,
        kwargs...
    ) where {F, P}
    primal = integrate(
        quadalg, adalg, f, domain, params;
        #    norm = norm,
        kwargs...
    )

    function integrate_pullbacks(Δ)
        function dfdp(x, params)
            _, back = Zygote.pullback(p -> f(x, p), params)
            back(Δ)[1]
        end
        ∂p = integrate(
            quadalg, adalg, dfdp, domain, params;
            #    norm = norm,
            kwargs...
        )
        # ∂lb = -f(lb,params)  #needs correct for dim > 1
        # ∂ub = f(ub,params)
        return nothing, nothing, nothing, nothing, nothing, ∂p
    end
    primal, integrate_pullbacks
end

# defines adjoint via ∫[f(x,p; ∂/∂p f(x,p)] dx, ie it fuses the primal, post the primal calculation
# has flag to only compute quad norm with respect to only the primal in the pull-back. Gives same quadrature points as doing forwarddiff
Zygote.@adjoint function integrate(
        quadalg, adalg::PostfusedAD, f::F, domain,
        params::P;
        #    norm = norm,
        kwargs...
    ) where {F, P}
    @assert f isa IntegralFunction{false} "adjoint doesn't support in-place or batching"
    primal = integrate(
        quadalg, adalg, f, domain, params;
        norm = norm,
        kwargs...
    )

    nout = length(primal)
    # _norm = adalg.norm_partials ? norm : primalnorm(nout, norm)

    function integrate_pullbacks(Δ)
        function dfdp(x, params)
            y, back = Zygote.pullback(p -> f(x, p), params)
            [y; back(Δ)[1]]   #TODO need to match proper array type? promote_type???
        end
        ∂p = integrate(
            quadalg, adalg, dfdp, domain, params;
            #    norm = _norm,
            kwargs...
        )
        return nothing, nothing, nothing, nothing, nothing, @view ∂p[(nout + 1):end]
    end
    primal, integrate_pullbacks
end

# Fuses primal and partials prior to pullback, I doubt this will stick around based on required system evals.
Zygote.@adjoint function integrate(
        quadalg, adalg::PrefusedAD, f::F, domain,
        params::P;
        #    norm = norm,
        kwargs...
    ) where {F, P}
    @assert f isa IntegralFunction{false} "adjoint doesn't support in-place or batching"
    # from Seth Axen via Slack
    # Does not work w/ ArrayPartition unless with following hack
    # Base.similar(A::ArrayPartition, ::Type{T}, dims::NTuple{N,Int}) where {T,N} = similar(Array(A), T, dims)
    # TODO add ArrayPartition similar fix upstream, see https://github.com/SciML/RecursiveArrayTools.jl/issues/135
    ∂f_∂params(x, params) = only(Zygote.jacobian(p -> f(x, p), params))
    f_augmented(x, params) = [f(x, params); ∂f_∂params(x, params)...] #TODO need to match proper array type? promote_type???
    nout = length(f(sum(domain) / 2, params))
    # _norm = adalg.norm_partials ? norm : primalnorm(nout, norm)

    res = integrate(
        quadalg, adalg, f_augmented, domain, params;
        # norm = _norm,
        kwargs...
    )
    primal = first(res)
    function integrate_pullback(Δy)
        ∂params = Δy .* conj.(@view(res[(nout + 1):end]))
        return nothing, nothing, nothing, nothing, nothing, ∂params
    end
    primal, integrate_pullback
end

# define norm function based only on primal part of fused integrand
function primalnorm(nout, fnorm)
    return x -> fnorm(@view x[1:nout])
end

"""
    centralmoment(n, exprob::ExpectationProblem, expalg::AbstractExpectationAlgorithm; kwargs...)

Compute the first `n` central moments of the observable defined in `exprob`.

Returns a vector `[μ₁, μ₂, ..., μₙ]` where `μₖ` is the k-th central moment.
Note that the first central moment is always 0 (by definition).

The central moments are computed using the binomial expansion:
`μₙ = E[(X - μ)ⁿ] = Σₖ₌₀ⁿ C(n,k) (-μ)^(n-k) E[X^k]`

where `μ = E[X]` is the mean.

## Arguments
- `n`: The highest order central moment to compute (n ≥ 1)
- `exprob`: An `ExpectationProblem` with a scalar-valued observable
- `expalg`: The algorithm to use (`Koopman()` or `MonteCarlo(trajectories)`)
- `kwargs...`: Additional keyword arguments passed to `solve`

## Example
```julia
using SciMLExpectations, Distributions

gd = GenericDistribution(Normal(0, 1))
g(u, p) = u[1]  # Identity function
exprob = ExpectationProblem(g, gd, nothing)

# Compute first 4 central moments
moments = centralmoment(4, exprob, Koopman())
# moments[1] ≈ 0 (1st central moment is always 0)
# moments[2] ≈ 1 (variance of N(0,1))
# moments[3] ≈ 0 (skewness of N(0,1))
# moments[4] ≈ 3 (kurtosis of N(0,1))
```
"""
function centralmoment(
        n::Int, exprob::ExpectationProblem, expalg::AbstractExpectationAlgorithm,
        args...; kwargs...
    )
    n >= 1 || throw(ArgumentError("n must be at least 1"))

    # Get the original observable
    g_orig = observable(exprob)

    # Create a new observable that returns [g(x)^1, g(x)^2, ..., g(x)^n]
    function g_powers(x, p)
        val = g_orig(x, p)
        return [val^i for i in 1:n]
    end

    # Create a new ExpectationProblem with the modified observable
    new_exprob = ExpectationProblem(exprob.S, g_powers, exprob.h, exprob.d, exprob.params)

    # Solve the expectation problem
    sol = solve(new_exprob, expalg, args...; kwargs...)

    # Extract raw moments E[X], E[X^2], ..., E[X^n]
    raw_moments = sol.u
    μ = raw_moments[1]  # E[X] = mean

    # Compute central moments using binomial expansion
    # μₙ = E[(X - μ)ⁿ] = Σₖ₌₀ⁿ C(n,k) (-μ)^(n-k) E[X^k]
    # where E[X^0] = 1
    central_moments = similar(raw_moments)

    for m in 1:n
        cm = zero(eltype(raw_moments))
        for k in 0:m
            # E[X^k] is raw_moments[k] for k >= 1, and 1 for k = 0
            E_Xk = k == 0 ? one(eltype(raw_moments)) : raw_moments[k]
            cm += binomial(m, k) * (-μ)^(m - k) * E_Xk
        end
        central_moments[m] = cm
    end

    return central_moments
end

"""
    centralmoment(n, exprob::ExpectationProblem{F}, expalg::AbstractExpectationAlgorithm; kwargs...) where {F <: Union{SystemMap, ProcessNoiseSystemMap}}

Compute the first `n` central moments for differential equation systems.

See `centralmoment(n, exprob, expalg)` for details.
"""
function centralmoment(
        n::Int, exprob::ExpectationProblem{F},
        expalg::AbstractExpectationAlgorithm, args...;
        kwargs...
    ) where {F <: Union{SystemMap, ProcessNoiseSystemMap}}
    n >= 1 || throw(ArgumentError("n must be at least 1"))

    # Get the original observable
    g_orig = observable(exprob)

    # Create a new observable that returns [g(sol,p)^1, g(sol,p)^2, ..., g(sol,p)^n]
    function g_powers(sol, p)
        val = g_orig(sol, p)
        return [val^i for i in 1:n]
    end

    # Create a new ExpectationProblem with the modified observable
    new_exprob = ExpectationProblem(exprob.S, g_powers, exprob.h, exprob.d, exprob.params)

    # Solve the expectation problem
    sol = solve(new_exprob, expalg, args...; kwargs...)

    # Extract raw moments E[X], E[X^2], ..., E[X^n]
    raw_moments = sol.u
    μ = raw_moments[1]  # E[X] = mean

    # Compute central moments using binomial expansion
    central_moments = similar(raw_moments)

    for m in 1:n
        cm = zero(eltype(raw_moments))
        for k in 0:m
            E_Xk = k == 0 ? one(eltype(raw_moments)) : raw_moments[k]
            cm += binomial(m, k) * (-μ)^(m - k) * E_Xk
        end
        central_moments[m] = cm
    end

    return central_moments
end
