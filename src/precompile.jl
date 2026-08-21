@setup_workload begin
    precompile_distribution = GenericDistribution(Distributions.Uniform(-1.0, 1.0))
    precompile_expectation_problem = ExpectationProblem(
        (x, p) -> x[1]^2, precompile_distribution, nothing
    )

    @compile_workload begin
        solve(
            precompile_expectation_problem, Koopman(); quadalg = HCubatureJL(),
            ireltol = 1.0e-3, iabstol = 1.0e-3
        )
    end
end
