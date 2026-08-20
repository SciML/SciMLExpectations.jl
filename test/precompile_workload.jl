using Test, SciMLExpectations, Distributions, Integrals

@testset "Precompile workload" begin
    distribution = GenericDistribution(Uniform(-1.0, 1.0))
    expectation_problem = ExpectationProblem((x, p) -> x[1]^2, distribution, nothing)
    solution = solve(
        expectation_problem, Koopman(); quadalg = HCubatureJL(),
        ireltol = 1.0e-6, iabstol = 1.0e-6
    )

    @test solution.u[1] ≈ 1 / 3 atol = 1.0e-6
end
