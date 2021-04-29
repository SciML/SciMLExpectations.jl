using SafeTestsets

# @time @safetestset "ProbInts tests" begin include("probints.jl") end
# @time @safetestset "Koopman Tests" begin include("koopman.jl") end
@safetestset  "Koopman Tests" begin include("koopman_stable.jl") end

