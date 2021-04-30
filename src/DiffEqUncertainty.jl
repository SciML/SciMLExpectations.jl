module DiffEqUncertainty

# LinearAlgebra
using DiffEqBase, Statistics, Reexport, RecursiveArrayTools,
    Distributions, KernelDensity, Zygote, LinearAlgebra, Random
@reexport using Quadrature

include("problem_types.jl")
include("probints.jl")
include("koopman.jl")

# Type Piracy, should upstream
Base.eltype(K::UnivariateKDE)  = eltype(K.density)
Base.minimum(K::UnivariateKDE) = minimum(K.x)
Base.maximum(K::UnivariateKDE) = maximum(K.x)

export ProbIntsUncertainty,AdaptiveProbIntsUncertainty
export expectation#, centralmoment
export Koopman, MonteCarlo
export PrefusedAD,PostfusedAD, NonfusedAD
export integrate, GenericDistribution, build_integrand, SystemMap, ExpectationProblem

end
