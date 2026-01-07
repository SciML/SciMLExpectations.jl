using SafeTestsets, Test

@testset "Integrals" begin
    @safetestset "Quality Assurance" include("qa.jl")
    @safetestset "JET static analysis" include("jet.jl")
    @safetestset "Expectation Process Noise Tests" include("processnoise.jl")
    @safetestset "Expectation Interface Tests" include("interface.jl")
    @safetestset "Expectation Solve Tests" include("solve.jl")
    @safetestset "Expectation Differentiation Tests" include("differentiation.jl")
end
