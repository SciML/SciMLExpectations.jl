using SafeTestsets, Test, Pkg

const GROUP = get(ENV, "GROUP", "All")

@testset "SciMLExpectations" begin
    if GROUP == "All" || GROUP == "Core"
        @safetestset "Expectation Process Noise Tests" include("processnoise.jl")
        @safetestset "Expectation Interface Tests" include("interface.jl")
        @safetestset "Expectation Solve Tests" include("solve.jl")
        @safetestset "Expectation Differentiation Tests" include("differentiation.jl")
    end

    if GROUP == "QA"
        Pkg.activate("qa")
        Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
        Pkg.instantiate()
        @safetestset "Quality Assurance" include("qa/qa.jl")
    end
end
