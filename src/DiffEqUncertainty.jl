module DiffEqUncertainty

# LinearAlgebra
using DiffEqBase, Statistics, Reexport, RecursiveArrayTools,
    Distributions, KernelDensity, Zygote, LinearAlgebra, Random
using Parameters: @unpack

import DiffEqBase: solve
# import Base: rand, maximum, minimum, extrema
# import Distributions: pdf

@reexport using Quadrature

include("system_utils.jl")
include("distribution_utils.jl")
include("problem_types.jl")
include("probints.jl")
include("koopman.jl")

# Type Piracy, should upstream
Base.eltype(K::UnivariateKDE)  = eltype(K.density)
Base.minimum(K::UnivariateKDE) = minimum(K.x)
Base.maximum(K::UnivariateKDE) = maximum(K.x)
Base.extrema(K::UnivariateKDE) = minimum(K), maximum(K)

Base.minimum(d::AbstractMvNormal) = fill(-Inf, length(d))
Base.maximum(d::AbstractMvNormal) = fill(Inf, length(d))
Base.extrema(d::AbstractMvNormal) = minimum(d), maximum(d)

Base.minimum(d::Product) = minimum.(d.v)
Base.maximum(d::Product) = maximum.(d.v)
Base.extrema(d::Product) = minimum(d), maximum(d)

export ProbIntsUncertainty,AdaptiveProbIntsUncertainty
export expectation#, centralmoment
export Koopman, MonteCarlo
export PrefusedAD,PostfusedAD, NonfusedAD
export integrate, GenericDistribution, build_integrand, SystemMap, ExpectationProblem

end
