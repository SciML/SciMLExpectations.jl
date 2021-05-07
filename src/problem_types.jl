
abstract type AbstractUncertaintyProblem end

struct ExpectationProblem{TS, TG, TH, TF, TP} <: AbstractUncertaintyProblem
    # ∫ g(S(h(x,p)))*f(x)dx
    S::TS  # mapping,                 S: 𝕐 × ℚ → 𝕐
    g::TG  # observable(output_func), g: 𝕐 × ℚ → ℝⁿᵒᵘᵗ  #TODO or should this be g: 𝕐 × ℙ → ℝⁿᵒᵘᵗ, requires update in build_integrand, if so
    h::TH  # cov(input_func),         h: 𝕏 × ℙ → 𝕐 × ℚ
    d::TF  # distribution,            pdf(d,x): 𝕏 → ℝ
    params::TP
    nout::Int
end 

function ExpectationProblem(g, pdist, params; nout = 1)
    h(x,u,p) = x, p
    S(x,p) = x
    ExpectationProblem(S, g, h, pdist, params, nout)
end

function ExpectationProblem(sm::SystemMap, g, h, d; nout = 1)
    ExpectationProblem(sm, g, h, d, 
        ArrayPartition(deepcopy(sm.prob.u0),deepcopy(sm.prob.p)),
        nout)
end

distribution(prob::ExpectationProblem) = prob.d
mapping(prob::ExpectationProblem) = prob.S
observable(prob::ExpectationProblem) = prob.g
input_cov(prob::ExpectationProblem) = prob.h
parameters(prob::ExpectationProblem) = prob.params

function build_integrand(prob::ExpectationProblem{F}) where F<:SystemMap
    @unpack S, g, h, d = prob
    function(x,p)
        ū, p̄ = h(x, p.x[1], p.x[2])
        g(S(ū,p̄), p̄)*pdf(d,x)   
    end
end

function build_integrand(prob::ExpectationProblem)
    @unpack g, d = prob
    function(x,p)
        g(x,p)*pdf(d,x)
    end
end


## 
# struct CentralMomentProblem
#     ns::NTuple{Int,N}
#     altype::Union{NestedExpectation, BinomialExpansion} #Should rely be in solve
#     exp_prob::ExpectationProblem
# end

