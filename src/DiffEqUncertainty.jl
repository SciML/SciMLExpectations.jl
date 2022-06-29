module DiffEqUncertainty

using DiffEqBase, Statistics, Distributions, Reexport
@reexport using Integrals
using KernelDensity

include("probints.jl")
include("koopman.jl")

# Type Piracy, should upstream
Base.eltype(K::UnivariateKDE)  = eltype(K.density)
Base.minimum(K::UnivariateKDE) = minimum(K.x)
Base.maximum(K::UnivariateKDE) = maximum(K.x)

export ProbIntsUncertainty,AdaptiveProbIntsUncertainty
export expectation, centralmoment, Koopman, MonteCarlo

end
