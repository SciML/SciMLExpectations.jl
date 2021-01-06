module DiffEqUncertainty

using DiffEqBase, Statistics, Distributions, Reexport
@reexport using Quadrature
using KernelDensity

include("probints.jl")
include("koopman.jl")

# Type Piracy, should upstream
Base.eltype(K::UnivariateKDE)  = eltype(K.density)
Base.minimum(K::UnivariateKDE) = minimum(K.density)
Base.maximum(K::UnivariateKDE) = maximum(K.density)

export ProbIntsUncertainty,AdaptiveProbIntsUncertainty
export expectation, centralmoment, Koopman, MonteCarlo

end
