using Test, TestExtras,
    DiffEqUncertainty, OrdinaryDiffEq, Distributions,
    StaticArrays, ComponentArrays, 
    ForwardDiff, FiniteDiff,Zygote

import DiffEqUncertainty: bounds, indices

## Array
function pend!(du, u, p, t)
    du[1] = u[2]
    du[2] = -p[1]/p[2]*sin(u[1])
    nothing
end
pendsa(u,p,t) = SVector(u[2], -p[1]/p[2]*sin(u[1]))

g(soln) = soln[1, end]
tspan = (0.0,10.0)

eoms = (pend!, pendsa, pend!)
u0s = ([π/2, 0.0], 
    SVector(π/2, 0.0),
    ComponentArray(state=(θ = π/2, θ̇ = 0.0)))
ps = ([9.807, 1.0],
    SVector(9.807, 1.0),
    ComponentArray(g=9.807, ℓ = 1.0))

@testset "Koopman Array/Tuple Interface Transform" begin
    @testset "Mixed Distributions" begin
        u_dist = [0.0, Uniform(-.1,.1)]
        u_pair = (2=>u_dist[2],)
        for u ∈ u0s
            @test DiffEqUncertainty.transform_interface(u, u_dist) == (map(zero, u), u_pair)
            @test DiffEqUncertainty.transform_interface(u, (u_dist...,)) == (map(zero, u), u_pair)
        end
    end

    @testset "No Distributions" begin
        u_dist = [0.0, 0.0]
        u_pair = ()
        for u ∈ u0s
            @test DiffEqUncertainty.transform_interface(u, u_dist) == (map(zero, u), u_pair)
            @test DiffEqUncertainty.transform_interface(u, (u_dist...,)) == (map(zero, u), u_pair)
        end
    end

    @testset "All Distributions" begin
        u_dist = [Normal(0.0,1.0), Uniform(-.1,.1)]
        u_pair = (1=>u_dist[1], 2=>u_dist[2])
        for u ∈ u0s
            @test DiffEqUncertainty.transform_interface(u, u_dist) == (map(zero, u), u_pair)
            @test DiffEqUncertainty.transform_interface(u, (u_dist...,)) == (map(zero, u), u_pair)
        end
    end

    @testset "Type Stability" begin
        u_dist = (0.0, Uniform(-.1,.1))
        for u ∈ u0s
            @constinferred DiffEqUncertainty.transform_interface(u, u_dist)
        end
    end
end

@testset "Koopman JointPdf Transform" begin
    udist = (1=> Uniform(-1,1), 3=>Normal(0,1))
    pdist = (1=> Uniform(5,6), )
    x = [mean.(last.(udist))...; mean.(last.(pdist))...]

    f = let udist = udist, pdist=pdist
            x-> prod(pdf(a,b) for (a,b) in zip((last.(udist)...,last.(pdist)...),x))
    end
    
    j= JointPdf(f, first.(udist), minimum.(last.(udist)), maximum.(last.(udist)), 
                                 first.(pdist), minimum.(last.(pdist)), maximum.(last.(pdist)))
    @constinferred JointPdf(f, first.(udist), minimum.(last.(udist)), maximum.(last.(udist)), 
                                 first.(pdist), minimum.(last.(pdist)), maximum.(last.(pdist)))
    j_pairs = JointPdf(udist,pdist)
    @constinferred JointPdf(udist,pdist)
    j_array = JointPdf([last(udist[1]), 1.0, last(udist[2])], [last(pdist[1]), 1.0, 2.0])
    j_tuple = JointPdf((last(udist[1]), 1.0, last(udist[2])), (last(pdist[1]), 1.0, 2.0))
    @constinferred JointPdf((last(udist[1]), 1.0, last(udist[2])), (last(pdist[1]), 1.0, 2.0))

    for _j in (j_pairs, j_array, j_tuple)
        @test bounds(j) == bounds(_j)
        @test indices(j) ==indices(_j)
        @test j(x) == _j(x)
        @constinferred _j(x)
    end
end


@testset "Koopman Expectation " begin
    u0_dist = (1 => Uniform(.9*π/4, 1.1*π/4),)
    ps_dist = (2 => Uniform(.9, 1.1), )
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
        ps = [9.807,1.0]
        tspan = (0.0,10.0)
        prob = ODEProblem{true}(pend!, u0, tspan, ps)
        u0_dist = (1 => Uniform(.9*π/4, 1.1*π/4),)
        ps_dist = (2 => Uniform(.9, 1.1), )
        expectation(g, prob, u0_dist, ps_dist, Koopman(), Tsit5())[1]
    end
    @testset "Correctness" begin
        fd = FiniteDiff.finite_difference_derivative(loss, 0.0)
        @test ForwardDiff.derivative(loss, 0.0) ≈ fd rtol=1e-2
    end
    @testset "Type Stability" begin
        pt = 0.0
        @constinferred ForwardDiff.derivative(loss, pt)
        cfg = ForwardDiff.GradientConfig(loss∘first, [pt, 0.0])  # required, as config heuristic is type unstable, see: https://juliadiff.org/ForwardDiff.jl/latest/user/advanced.html#Configuring-Chunk-Size-1
        @constinferred ForwardDiff.gradient(loss∘first,[pt, 0.0], cfg)
    end
end
