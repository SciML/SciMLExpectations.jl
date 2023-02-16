module SciMLExpectations

# LinearAlgebra
using DiffEqBase, SciMLBase, Statistics, Reexport, RecursiveArrayTools,
      Distributions, KernelDensity, Zygote, LinearAlgebra, Random
using Parameters: @unpack

@reexport using Integrals
import DiffEqBase: solve

include("system_utils.jl")
include("distribution_utils.jl")
include("problem_types.jl")
include("solution_types.jl")
include("expectation.jl")

# Type Piracy, should upstream
Base.eltype(K::UnivariateKDE) = eltype(K.density)
Base.minimum(K::UnivariateKDE) = minimum(K.x)
Base.maximum(K::UnivariateKDE) = maximum(K.x)
Base.extrema(K::UnivariateKDE) = minimum(K), maximum(K)

export Koopman, MonteCarlo, PrefusedAD, PostfusedAD, NonfusedAD
export GenericDistribution, SystemMap, ExpectationProblem, build_integrand

end
