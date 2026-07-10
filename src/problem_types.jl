abstract type AbstractUncertaintyProblem end

"""
    ExpectationProblem(S, g, h, d, params)
    ExpectationProblem(g, d, params; nout = nothing)
    ExpectationProblem(sm::SystemMap, g, h, d; nout = nothing)
    ExpectationProblem(sm::ProcessNoiseSystemMap, g, h; nout = nothing)

Represent an expectation of an observable over an uncertainty distribution.

An `ExpectationProblem` defines the data needed to compute an integral of the form
`integral g(S(u, p), p) * pdf(d, x) dx`, where `h(x, u0, p)` maps uncertain
inputs into initial conditions and parameters for the system map `S`. The
function-only constructor uses identity maps for `S` and `h`.

## Arguments

  - `S`: System map called as `S(u, p)`.
  - `g`: Observable called as `g(u, p)` for function problems or `g(sol, p)` for
    system maps.
  - `h`: Covariate map called as `h(x, u0, p)`.
  - `d`: Distribution of uncertain inputs. It must support the operations needed by
    the chosen expectation algorithm, such as `pdf`, `rand`, and `extrema`.
  - `params`: Parameters passed to the observable and integration problem.
  - `sm`: A `SystemMap` or `ProcessNoiseSystemMap`.
  - `nout`: Deprecated and unused.

## Fields

  - `S`: Stored system map.
  - `g`: Stored observable.
  - `h`: Stored covariate map.
  - `d`: Stored uncertainty distribution.
  - `params`: Stored parameters.

## Returns

An `ExpectationProblem` that can be solved with `solve(prob, Koopman())` or
`solve(prob, MonteCarlo(trajectories))`.
"""
struct ExpectationProblem{TS, TG, TH, TF, TP} <: AbstractUncertaintyProblem
    # defines ∫ g(S(h(x,u0,p)))*f(x)dx
    # 𝕏 = uncertainty space, 𝕌 = Initial condition space, ℙ = model parameter space,
    S::TS  # mapping,                 S: 𝕌 × ℙ → 𝕌
    g::TG  # observable(output_func), g: 𝕌 × ℙ → ℝⁿᵒᵘᵗ
    h::TH  # cov(input_func),         h: 𝕏 × 𝕌 × ℙ → 𝕌 × ℙ
    d::TF  # distribution,            pdf(d,x): 𝕏 → ℝ
    params::TP
end

# Constructor for general maps/functions
function ExpectationProblem(g, pdist, params; nout = nothing)
    !isnothing(nout) && @warn "nout is deprecated and unused"
    h(x, u, p) = x, p
    S(x, p) = x
    return ExpectationProblem(S, g, h, pdist, params)
end

# Constructor for DEProblems
function ExpectationProblem(sm::SystemMap, g, h, d; nout = nothing)
    !isnothing(nout) && @warn "nout is deprecated and unused"
    return ExpectationProblem(
        sm, g, h, d,
        ArrayPartition(deepcopy(sm.prob.u0), deepcopy(sm.prob.p))
    )
end

distribution(prob::ExpectationProblem) = prob.d
mapping(prob::ExpectationProblem) = prob.S
observable(prob::ExpectationProblem) = prob.g
input_cov(prob::ExpectationProblem) = prob.h
parameters(prob::ExpectationProblem) = prob.params

##
# struct CentralMomentProblem
#     ns::NTuple{Int,N}
#     altype::Union{NestedExpectation, BinomialExpansion} #Should rely be in solve
#     exp_prob::ExpectationProblem
# end

function ExpectationProblem(sm::ProcessNoiseSystemMap, g, h; nout = nothing)
    !isnothing(nout) && @warn "nout is deprecated and unused"
    d = GenericDistribution((Truncated(Normal(), -4.0, 4.0) for i in 1:(sm.n))...)
    return ExpectationProblem(sm, g, h, d, deepcopy(sm.prob.p))
end
