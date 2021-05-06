using Test, TestExtras,
    DiffEqUncertainty, OrdinaryDiffEq, Distributions,
    StaticArrays, ComponentArrays, 
    ForwardDiff, FiniteDiff, Zygote, Random

function pend!(du, u, p, t)
    du[1] = u[2]
    du[2] = -p[1]/p[2]*sin(u[1])
    nothing
end
function pendca!(du, u, p, t)
    du.state.θ = u.state.θ̇
    du.state.θ̇ = -p.g/p.ℓ*sin(u.state.θ)
end
pendsa(u,p,t) = SVector(u[2], -p[1]/p[2]*sin(u[1]))

g(soln) = soln[1, end]
tspan = (0.0,10.0)

eoms = (pend!, pendsa, pend!, pendca!)
u0s = ([π/2, 0.0], 
    SVector(π/2, 0.0),
    ComponentArray(state=(θ = π/2, θ̇ = 0.0)),
    ComponentArray(state=(θ = π/2, θ̇ = 0.0)))
ps = ([9.807, 1.0],
    SVector(9.807, 1.0),
    ComponentArray(g=9.807, ℓ = 1.0),
    ComponentArray(g=9.807, ℓ = 1.0))

@testset "GenericDistribution" begin
    dists = (Uniform(1,2), Uniform(3,4), Normal(0,1))
    x = [mean(d) for d in dists]

    pdf_f = let dists = dists
        x->mapreduce(pdf, *, dists, x)
    end
    rand_f = let dists = dists; 
        () -> [rand(d) for d in dists]; 
    end
    lb = tuple(minimum.(dists)...)
    ub = tuple(maximum.(dists)...)

    P = Product([dists...])
    gd_ind = @constinferred GenericDistribution(dists...)
    gd_raw = @constinferred GenericDistribution(pdf_f, rand_f, lb, ub)
    @constinferred GenericDistribution(pdf_f, rand_f, [lb...], [ub...])

    for gd in (gd_ind, gd_raw)
        @test minimum(gd) == tuple(minimum(P)...)
        @test maximum(gd) == tuple(maximum(P)...)
        @test extrema(gd) == (tuple(minimum(P)...), tuple(maximum(P)...))
        @test pdf(gd,x) ≈ pdf(P,x)
        @constinferred pdf(gd,x)
        
        Random.seed!(0)
        @test rand(gd) == begin 
            Random.seed!(0); 
            rand(P)
        end
        @constinferred rand(gd)
    end
end

@testset "SystemMap" begin
    for (f,u,p) ∈ zip(eoms, u0s, ps)
        prob = ODEProblem(f, u, tspan, p)        
        sm = @constinferred SystemMap(prob, Tsit5(); saveat=1.0)
        sm_soln = @constinferred sm(u,p)
        soln = solve(prob, Tsit5(); saveat=1.0)
        @test sm_soln.t == soln.t
        @test sm_soln.u == soln.u
    end
end

# @testset "Koopman Expectation " begin
#     u0_dists = (
#                  (1 => Uniform(.9*π/4, 1.1*π/4),),
#                  (Uniform(.9*π/4, 1.1*π/4), 0.0)
#     )
#     ps_dists = (
#                  (2 => Uniform(.9, 1.1), ),
#                  (Uniform(.9, 1.1), 1.0)
#     )
#     @testset "Type Stability" begin
#         for (u0_dist, ps_dist) ∈ zip(u0_dists, ps_dists)
#             for (f,u,p) ∈ zip(eoms, u0s, ps)
#                 prob = ODEProblem(f, u, tspan, p)        
#                 @constinferred expectation(g, prob, u0_dist, ps_dist, Koopman(), Tsit5())
#             end
#         end
#     end
# end

# @testset "Koopman Expectation AD" begin
#     function loss(x::T) where {T<:Real}
#         u0 = [0.0, x]
#         ps = [9.807,1.0]
#         tspan = (0.0,10.0)
#         prob = ODEProblem{true}(pend!, u0, tspan, ps)
#         u0_dist = (1 => Uniform(.9*π/4, 1.1*π/4),)
#         ps_dist = (2 => Uniform(.9, 1.1), )
#         expectation(g, prob, u0_dist, ps_dist, Koopman(), Tsit5())[1]
#     end
#     @testset "Correctness" begin
#         fd = FiniteDiff.finite_difference_derivative(loss, 0.0)
#         @test ForwardDiff.derivative(loss, 0.0) ≈ fd rtol=1e-2
#     end
#     @testset "Type Stability" begin
#         pt = 0.0
#         @constinferred ForwardDiff.derivative(loss, pt)
#         cfg = ForwardDiff.GradientConfig(loss∘first, [pt, 0.0])  # required, as config heuristic is type unstable, see: https://juliadiff.org/ForwardDiff.jl/latest/user/advanced.html#Configuring-Chunk-Size-1
#         @constinferred ForwardDiff.gradient(loss∘first,[pt, 0.0], cfg)
#     end
# end
