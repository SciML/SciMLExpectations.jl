module SciMLExpectations

import DiffEqBase
import DiffEqBase: solve
import DiffEqNoiseProcess
import Distributions
import Distributions: Normal, Sampleable, Truncated, logpdf, pdf
import Integrals: HCubatureJL
import LinearAlgebra: Adjoint, norm
using Parameters: @unpack
import RecursiveArrayTools: ArrayPartition
import SciMLBase
import SciMLBase: BatchIntegralFunction, EnsembleProblem, EnsembleThreads,
    IntegralFunction, IntegralProblem, remake
import StaticArrays: SVector
import Statistics: mean
import Zygote
import ZygoteRules: @adjoint
using PrecompileTools: @compile_workload, @setup_workload

include("system_utils.jl")
include("distribution_utils.jl")
include("problem_types.jl")
include("solution_types.jl")
include("expectation.jl")
include("precompile.jl")

export Koopman, MonteCarlo, PrefusedAD, PostfusedAD, NonfusedAD
export GenericDistribution, SystemMap, ProcessNoiseSystemMap, ExpectationProblem,
    build_integrand, centralmoment

end
