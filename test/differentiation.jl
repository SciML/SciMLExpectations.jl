using Test, TestExtras,
      SciMLExpectations, OrdinaryDiffEq, Distributions,
      StaticArrays, ComponentArrays, Random, FiniteDiff,
      ForwardDiff, RecursiveArrayTools

include("setup.jl")

@testset "Koopman Expectation AD" begin
    function loss(x::T) where {T <: Real}
        u0 = [0.0, x]
        ps = [9.807, 1.0]
        tspan = (0.0, 10.0)
        prob = ODEProblem{true}(pend!, u0, tspan, ps)

        sm = SystemMap(prob, Tsit5(); saveat = 1.0, abstol = 1e-10, reltol = 1e-10)

        h(x, u, p) = u, p
        g(soln, p) = soln
        gd = product_distribution(Uniform(9, 11), Uniform(0.9 * π / 4, 1.1 * π / 4))
        ep = ExpectationProblem(sm, g, h, gd)
        solve(ep, Koopman()).u
    end
    @testset "Correctness" begin
        fd = FiniteDiff.finite_difference_derivative(loss, 0.0)
        @test ForwardDiff.derivative(loss, 0.0)≈fd rtol=1e-3
    end
    @testset "Type Stability" begin
        pt = 0.0
        #@test_broken @constinferred ForwardDiff.derivative(loss, pt)
        #cfg = ForwardDiff.GradientConfig(loss∘first, [pt, 0.0])  # required, as config heuristic is type unstable, see: https://juliadiff.org/ForwardDiff.jl/latest/user/advanced.html#Configuring-Chunk-Size-1
        #@test_broken @constinferred ForwardDiff.gradient(loss∘first,[pt, 0.0], cfg)
    end
end
