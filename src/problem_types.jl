abstract type AbstractUncertaintyProblem end

"""
```julia
ExpectationProblem(S, g, h, pdist, params)
ExpectationProblem(g, pdist, params)
ExpectationProblem(sm::SystemMap, g, h, d)
```

Defines ∫ g(S(h(x,u0,p)))*f(x)dx

## Arguments

Let 𝕏 = uncertainty space, 𝕌 = Initial condition space, ℙ = model parameter space

  - S: 𝕌 × ℙ → 𝕌 also known as system map.
  - g: 𝕌 × ℙ → ℝⁿᵒᵘᵗ also known as the observables or output function.
  - h: 𝕏 × 𝕌 × ℙ → 𝕌 × ℙ also known as covariate function.
  - pdf(d,x): 𝕏 → ℝ the uncertainty distribution of the initial states.
  - params
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
    ExpectationProblem(S, g, h, pdist, params)
end

# Constructor for DEProblems
function ExpectationProblem(sm::SystemMap, g, h, d; nout = nothing)
    !isnothing(nout) && @warn "nout is deprecated and unused"
    ExpectationProblem(sm, g, h, d,
        ArrayPartition(deepcopy(sm.prob.u0), deepcopy(sm.prob.p)))
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
    ExpectationProblem(sm, g, h, d, deepcopy(sm.prob.p))
end
