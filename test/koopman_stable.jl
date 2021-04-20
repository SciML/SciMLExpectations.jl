using Test, TestExtras,
    DiffEqUncertainty, OrdinaryDiffEq, Distributions,
    StaticArrays, ComponentArrays, 
    ForwardDiff, FiniteDiff

## Array
function pend!(du, u, p, t)
    du[1] = u[2]
    du[2] = -p[1]/p[2]*sin(u[1])
    nothing
end
pendsa(u,p,t) = SVector(u[2], -p[1]/p[2]*sin(u[1]))

g(soln) = soln[1, end]
tspan = (0.0,10.0)
u0_dist = (1 => Uniform(.9*π/2, 1.1*π/2),)
ps_dist = (2 => Uniform(.9, 1.1), )

@testset "Koopman Expectation " begin
    eoms = (pend!, pendsa, pend!)
    u0s = ([π/2, 0.0], 
        SVector(π/2, 0.0),
        ComponentArray(state=(θ = π/2, θ̇ = 0.0)))
    ps = ([9.807, 1.0],
        SVector(9.807, 1.0),
        ComponentArray(g=9.807, ℓ = 1.0))
    @testset "Type Stability" begin
        for (f,u,p) ∈ zip(eoms, u0s, ps)
            prob = ODEProblem(f, u, tspan, p)
            @constinferred expectation(g, prob, u0_dist, ps_dist, Koopman(), Tsit5())
        end
    end
end

@testset "Koopman Expectation AD" begin
    function loss(x::T) where {T<:Real}
        u0 = [0.0, x]
        ps = [9.807, 1.0]
        tspan = (0.0,10.0)
        prob = ODEProblem{true}(pend!, u0, tspan, ps)
        u0_dist = (1 => Uniform(.9*π/4, 1.1*π/4),)
        ps_dist = (2 => Uniform(.9, 1.1), )
        expectation(g, prob, u0_dist, ps_dist, Koopman(), Tsit5()).u
    end
    @testset "Correctness" begin
        fd = FiniteDiff.finite_difference_derivative(loss, 0.0)
        @test ForwardDiff.derivative(loss, 0.0) ≈ fd rtol=1e-2
    end
    @testset "Type Stability" begin
        @constinferred ForwardDiff.derivative(loss, 0.0)
    end
end
