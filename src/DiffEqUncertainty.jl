module DiffEqUncertainty

# LinearAlgebra
using DiffEqBase, Statistics, Reexport, RecursiveArrayTools, StaticArrays,
      Distributions, KernelDensity, Zygote, LinearAlgebra, Random
using Parameters: @unpack

@reexport using Integrals
import DiffEqBase: solve

include("expectation/system_utils.jl")
include("expectation/distribution_utils.jl")
include("expectation/problem_types.jl")
include("expectation/expectation.jl")

include("probints.jl")

# Type Piracy, should upstream
Base.eltype(K::UnivariateKDE) = eltype(K.density)
Base.minimum(K::UnivariateKDE) = minimum(K.x)
Base.maximum(K::UnivariateKDE) = maximum(K.x)
Base.extrema(K::UnivariateKDE) = minimum(K), maximum(K)

Base.minimum(d::AbstractMvNormal) = fill(-Inf, length(d))
Base.maximum(d::AbstractMvNormal) = fill(Inf, length(d))
Base.extrema(d::AbstractMvNormal) = minimum(d), maximum(d)

Base.minimum(d::Product) = minimum.(d.v)
Base.maximum(d::Product) = maximum.(d.v)
Base.extrema(d::Product) = minimum(d), maximum(d)

export ProbIntsUncertainty, AdaptiveProbIntsUncertainty

export Koopman, MonteCarlo, PrefusedAD, PostfusedAD, NonfusedAD
export GenericDistribution, SystemMap, ExpectationProblem, build_integrand

end
