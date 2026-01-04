using Test, TestExtras,
    SciMLExpectations, OrdinaryDiffEq, Distributions,
    Integrals, Cubature, Cuba,
    StaticArrays, ComponentArrays, Random

quadalgs = [
    HCubatureJL(),
    CubatureJLh(),
    CubatureJLp(),
    CubaSUAVE(),
    CubaDivonne(),
    CubaCuhre(),
]
quadalgs_batch = [CubatureJLh(), CubatureJLp(), CubaSUAVE(), CubaDivonne(), CubaCuhre()]
# batchmode = [EnsembleSerial(), EnsembleThreads()]#, EnsembleGPUArray()]

@testset "DiffEq Expectation Solve" begin
    function eom!(du, u, p, t, A)
        @inbounds begin
            du .= A * u
        end
        nothing
    end

    u0 = [1.0, 1.0]
    tspan = (0.0, 3.0)
    p = [1.0; 2.0]
    A = [0.0 1.0; -p[1] -p[2]]
    prob = ODEProblem((du, u, p, t) -> eom!(du, u, p, t, A), u0, tspan, p)
    u0s_dist = (Uniform(1, 10), truncated(Normal(3.0, 1), 0.0, 6.0))
    gd = GenericDistribution(u0s_dist...)
    cov(x, u, p) = x, p

    sm = SystemMap(prob, Tsit5(), save_everystep = false)

    analytical = (exp(A * tspan[end]) * [mean(d) for d in u0s_dist])
    @testset "Scalar Observable (nout = 1)" begin
        g(sol, p) = sol[1, end]
        exprob = ExpectationProblem(sm, g, cov, gd)
        for alg in quadalgs
            @test solve(
                exprob, Koopman(); quadalg = alg,
                ireltol = 1.0e-3, iabstol = 1.0e-3
            ).u[1] ≈ analytical[1] rtol = 1.0e-2
            # @constinferred solve(exprob, Koopman(); quadalg = alg)[1]  # Commented b/c no "broken" inferred macros and is not stable due to Quadrature.jl
            if alg ∈ quadalgs_batch
                s = solve(
                    exprob, Koopman(); quadalg = alg, ireltol = 1.0e-3, iabstol = 1.0e-3,
                    batch = 20
                ).u[1]
                @test s ≈ analytical[1] rtol = 1.0e-2
                # @constinferred solve(exprob, Koopman(); quadalg = alg, batch = 5)[1]  # Commented b/c no "broken" inferred macros and is not stable due to Quadrature.jl
            end
        end
        @test solve(exprob, MonteCarlo(10000)).u[1] ≈ analytical[1] rtol = 1.0e-2
        @constinferred solve(exprob, MonteCarlo(10000))
    end
    @testset "Vector-Valued Observable (nout > 1)" begin
        g(sol, p) = sol[:, end]
        exprob = ExpectationProblem(sm, g, cov, gd)
        for alg in quadalgs
            @test solve(
                exprob, Koopman(); quadalg = alg, ireltol = 1.0e-3,
                iabstol = 1.0e-3
            ).u ≈ analytical rtol = 1.0e-2
            # @constinferred solve(exprob, Koopman(); quadalg = alg)   # Commented b/c no "broken" inferred macros and is not stable due to Quadrature.jl
            if alg ∈ quadalgs_batch
                s = solve(
                    exprob, Koopman(); quadalg = alg, ireltol = 1.0e-3, iabstol = 1.0e-3,
                    batch = 20
                ).u[1]
                @test s ≈ analytical[1] rtol = 1.0e-2
                # @constinferred solve(exprob, Koopman(); quadalg = alg, batch = 5)[1]  # Commented b/c no "broken" inferred macros and is not stable due to Quadrature.jl
            end
        end
        @test solve(exprob, MonteCarlo(10000)).u ≈ analytical rtol = 1.0e-2
        @constinferred solve(exprob, MonteCarlo(10000))
    end
end

@testset "General Map Expectation Solve" begin
    gd = GenericDistribution(Uniform(0, 1), truncated(Normal(0, 1), -4, 4))
    p = [1.0, 2.0, 3.0]
    @testset "Scalar Observable (nout = 1)" begin
        g(u, p) = sum(p .* sin.(u[1])) + cos(u[2])
        analytical = 2 * sin(1 / 2)^2 * sum(p) + 1 / sqrt(exp(1))
        exprob = ExpectationProblem(g, gd, p)
        for alg in quadalgs
            @test solve(exprob, Koopman(); quadalg = alg).u[1] ≈ analytical rtol = 1.0e-2
        end
        @test solve(exprob, MonteCarlo(10000)).u[1] ≈ analytical rtol = 1.0e-1
    end
    @testset "Vector-Valued Observable (nout > 1)" begin
        g(u, p) = [sum(p .* sin.(u[1])) + cos(u[2]), cos(u[2])]
        analytical = [2 * sin(1 / 2)^2 * sum(p) + 1 / sqrt(exp(1)), 1 / sqrt(exp(1))]
        exprob = ExpectationProblem(g, gd, p)
        for alg in quadalgs
            @test solve(exprob, Koopman(); quadalg = alg).u ≈ analytical rtol = 1.0e-2
        end
        @test solve(exprob, MonteCarlo(10000)).u ≈ analytical rtol = 1.0e-1
    end
end

@testset "centralmoment Tests" begin
    @testset "General Map" begin
        # Test case: Standard Normal distribution
        # Central moments of N(0,1):
        # μ₁ = 0 (1st central moment is always 0)
        # μ₂ = 1 (variance)
        # μ₃ = 0 (skewness for symmetric distribution)
        # μ₄ = 3 (kurtosis)

        gd = GenericDistribution(Normal(0, 1))
        g(u, p) = u[1]  # Identity function
        exprob = ExpectationProblem(g, gd, nothing)

        # Test with Koopman
        moments_k = centralmoment(4, exprob, Koopman())
        @test abs(moments_k[1]) < 0.01  # μ₁ ≈ 0
        @test abs(moments_k[2] - 1) < 0.01  # μ₂ ≈ 1
        @test abs(moments_k[3]) < 0.01  # μ₃ ≈ 0
        @test abs(moments_k[4] - 3) < 0.1  # μ₄ ≈ 3

        # Test with MonteCarlo
        moments_m = centralmoment(4, exprob, MonteCarlo(100000))
        @test abs(moments_m[1]) < 0.01  # μ₁ ≈ 0
        @test abs(moments_m[2] - 1) < 0.1  # μ₂ ≈ 1 (looser tolerance for MC)
        @test abs(moments_m[3]) < 0.2  # μ₃ ≈ 0
        @test abs(moments_m[4] - 3) < 0.5  # μ₄ ≈ 3
    end

    @testset "SystemMap" begin
        # Test case: Simple linear ODE with uncertain initial condition
        # du/dt = -0.5 * u, u(0) ~ truncated Normal
        # Solution: u(t) = u0 * exp(-0.5 * t)

        function f(du, u, p, t)
            du[1] = -0.5 * u[1]
            nothing
        end

        u0 = [1.0]
        tspan = (0.0, 2.0)
        p = nothing
        prob = ODEProblem(f, u0, tspan, p)

        u0_dist = truncated(Normal(1.0, 0.1), 0.5, 1.5)
        gd = GenericDistribution(u0_dist)

        cov(x, u, p) = [x[1]], p
        g(sol, p) = sol[1, end]  # Observable: final value

        sm = SystemMap(prob, Tsit5(), save_everystep = false)
        exprob = ExpectationProblem(sm, g, cov, gd)

        # Expected variance: Var(u0) * exp(-2)
        expected_var = var(u0_dist) * exp(-2 * 1)

        # Test with Koopman
        moments_k = centralmoment(2, exprob, Koopman())
        @test abs(moments_k[1]) < 1.0e-10  # μ₁ should be exactly 0
        @test abs(moments_k[2] - expected_var) < 0.001

        # Test with MonteCarlo
        moments_m = centralmoment(2, exprob, MonteCarlo(10000))
        @test abs(moments_m[1]) < 1.0e-10  # μ₁ should be exactly 0
        @test abs(moments_m[2] - expected_var) < 0.01
    end
end
