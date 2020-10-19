using Test

@time @testset "ProbInts tests" begin include("probints.jl") end
@time @testset "Problem Tests" begin include("problems.jl") end
@time @testset "Koopman Tests" begin include("koopman.jl") end
