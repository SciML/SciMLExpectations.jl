abstract type AbstractExpectationAlgorithm <: DiffEqBase.DEAlgorithm end

"""
```julia
Koopman()
```
"""
struct Koopman
end


"""
```julia
MonteCarlo(trajectories::Int)
```
"""
struct MonteCarlo <: AbstractExpectationAlgorithm
    trajectories::Int
end

# Builds integrand for arbitrary functions
function build_integrand(prob::ExpectationProblem, ::Koopman, ::Val{false})
    @unpack g, d = prob
    function (x, p)
        g(x, p) * pdf(d, x)
    end
end

# Builds integrand for DEProblems
function build_integrand(prob::ExpectationProblem{F}, ::Koopman,
                         ::Val{false}) where {F <: SystemMap}
    @unpack S, g, h, d = prob
    function (x, p)
        ū, p̄ = h(x, p.x[1], p.x[2])
        g(S(ū, p̄), p̄) * pdf(d, x)
    end
end

function _make_view(x::Union{Vector{T}, Adjoint{T, Vector{T}}}, i) where {T}
    @view x[i]
end

function _make_view(x, i)
    @view x[:, i]
end

function build_integrand(prob::ExpectationProblem{F}, ::Koopman,
                         ::Val{true}) where {F <: SystemMap}
    @unpack S, g, h, d = prob

    if prob.nout == 1 # TODO fix upstream in quadrature, expected sizes depend on quadrature method is requires different copying based on nout > 1
        set_result! = @inline function (dx, sol)
            dx[:] .= sol[:]
        end
    else
        set_result! = @inline function (dx, sol)
            dx .= reshape(sol[:, :], size(dx))
        end
    end

    prob_func = function (prob, i, repeat, x)  # TODO is it better to make prob/output funcs outside of integrand, then call w/ closure?
        u0, p = h((_make_view(x, i)), prob.u0, prob.p)
        remake(prob, u0 = u0, p = p)
    end

    output_func(sol, i, x) = (g(sol, sol.prob.p) * pdf(d, (_make_view(x, i))), false)

    function (dx, x, p) where {T}
        trajectories = size(x, 2)
        # TODO How to inject ensemble method in solve? currently in SystemMap, but does that make sense?
        ensprob = EnsembleProblem(S.prob; output_func = (sol, i) -> output_func(sol, i, x),
                                  prob_func = (prob, i, repeat) -> prob_func(prob, i,
                                                                             repeat, x))
        sol = solve(ensprob, S.args...; trajectories = trajectories, S.kwargs...)
        set_result!(dx, sol)
        nothing
    end
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
    ExpectationSolution(mean(g(rand(dist), params) for _ in 1:(expalg.trajectories)),
                        nothing, nothing)
end

function DiffEqBase.solve(exprob::ExpectationProblem{F},
                          expalg::MonteCarlo) where {F <: SystemMap}
    d = distribution(exprob)
    cov = input_cov(exprob)
    S = mapping(exprob)
    g = observable(exprob)

    prob_func = function (prob, i, repeat)
        u0, p = cov(rand(d), prob.u0, prob.p)
        remake(prob, u0 = u0, p = p)
    end

    output_func(sol, i) = (g(sol, sol.prob.p), false)

    monte_prob = EnsembleProblem(S.prob;
                                 output_func = output_func,
                                 prob_func = prob_func)
    sol = solve(monte_prob, S.args...; trajectories = expalg.trajectories, S.kwargs...)
    ExpectationSolution(mean(sol.u), nothing, nothing)
end


"""
```julia
solve(exprob::ExpectationProblem, expalg::Koopman;
      maxiters = 1000000, batch = 0 , quadalg = HCubatureJL(),
      ireltol = 1e-2, iabstol = 1e-2, kwargs...)
```
Solve an `ExpectationProblem` using Koopman integration.
"""
function DiffEqBase.solve(prob::ExpectationProblem, expalg::Koopman, args...;
                          maxiters = 1000000,
                          batch = 0,
                          quadalg = HCubatureJL(),
                          ireltol = 1e-2, iabstol = 1e-2,
                          kwargs...)
    integrand = build_integrand(prob, expalg, Val(batch > 1))
    lb, ub = extrema(prob.d)

    sol = integrate(quadalg, integrand, lb, ub, prob.params;
                    reltol = ireltol, abstol = iabstol, maxiters = maxiters,
                    nout = prob.nout, batch = batch,
                    kwargs...)

    return ExpectationSolution(sol.u, sol.resid, sol)
end

# Integrate function to test new Adjoints, will need to roll up to Integrals.jl
function integrate(quadalg, f, lb::TB, ub::TB, p;
                   nout = 1, batch = 0,
                   kwargs...) where {TB}
    #TODO check batch iip type stability w/ IntegralProblem{XXXX}
    prob = IntegralProblem{batch > 1}(f, lb, ub, p; nout = nout, batch = batch)
    solve(prob, quadalg; kwargs...)
end
