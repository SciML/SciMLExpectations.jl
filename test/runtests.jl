using SafeTestsets

@safetestset "Expectation Interface Tests" begin include("interface.jl") end
@safetestset "Expectation Solve Tests" begin include("solve.jl") end
@safetestset "Expectation Differentiation Tests" begin include("differentiation.jl") end
