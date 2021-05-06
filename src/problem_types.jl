
abstract type AbstractUncertaintyProblem end

struct ExpectationProblem{TS, TG, TH, TF, TP} <: AbstractUncertaintyProblem
    # ∫ g(S(h(x,p)))*f(x)dx
    S::TS  # mapping,                 S: 𝕐 × ℚ → 𝕐
    g::TG  # observable(output_func), g: 𝕐 → ℝ
    h::TH  # cov(input_func),         h: 𝕏 × ℙ → 𝕐 × ℚ
    d::TF  # distribution,            pdf(d,x): 𝕏 → ℝ
    params::TP
end 

function ExpectationProblem(S, pdist, params)
    g(x) = x
    h(x,u,p) = x,p
    ExpectationProblem(S, g, h, pdist, ArrayPartition(eltype(params)[], params))
end

ExpectationProblem(sm::SystemMap, g, h, d) = 
    ExpectationProblem(sm, g, h, d, ArrayPartition(deepcopy(sm.prob.u0),deepcopy(sm.prob.p)))

distribution(prob::ExpectationProblem) = prob.d
mapping(prob::ExpectationProblem) = prob.S
observable(prob::ExpectationProblem) = prob.g
input_cov(prob::ExpectationProblem) = prob.h
parameters(prob::ExpectationProblem) = prob.params

function build_integrand(prob::ExpectationProblem)
    @unpack S, g, h, d = prob
    function(x,p)
        g(S(h(x, p.x[1], p.x[2])...))*pdf(d,x)
    end
end

## 
# struct CentralMomentProblem
#     ns::NTuple{Int,N}
#     altype::Union{NestedExpectation, BinomialExpansion} #Should rely be in solve
#     exp_prob::ExpectationProblem
# end

