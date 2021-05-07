using SafeTestsets

#TODO reenable tests
# @time @safetestset "ProbInts tests" begin include("probints.jl") end
@safetestset "Expectation Interface Tests" begin include("expectation/interface.jl") end
@safetestset "Expectation Solve Tests" begin include("expectation/solve.jl") end
