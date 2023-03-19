using SafeTestsets

@safetestset "Expectation Process Noise Tests" begin include("processnoise.jl") end
@safetestset "Expectation Interface Tests" begin include("interface.jl") end
@safetestset "Expectation Solve Tests" begin include("solve.jl") end
@safetestset "Expectation Differentiation Tests" begin include("differentiation.jl") end
