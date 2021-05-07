using Test, TestExtras,
    DiffEqUncertainty, OrdinaryDiffEq, Distributions,
    StaticArrays, ComponentArrays, Random

const DEU = DiffEqUncertainty
include("setup.jl")

@testset "GenericDistribution" begin
    dists = (Uniform(1,2), Uniform(3,4), Normal(0,1))
    x = [mean(d) for d in dists]

    pdf_f = let dists = dists
        x->mapreduce(pdf, *, dists, x)
    end
    rand_f = let dists = dists; 
        () -> [rand(d) for d in dists]; 
    end
    lb = [minimum.(dists)...]
    ub = [maximum.(dists)...]

    P = Product([dists...])
    gd_ind = @constinferred GenericDistribution(dists...)
    gd_raw = @constinferred GenericDistribution(pdf_f, rand_f, lb, ub)
    @constinferred GenericDistribution(pdf_f, rand_f, lb, ub)

    for gd in (gd_ind, gd_raw)
        @test minimum(gd) == minimum(P)
        @test maximum(gd) == maximum(P)
        @test extrema(gd) == extrema(P)
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

@testset "ExpectationProblem" begin
    @testset "Interface" begin
        getters = (DEU.distribution, DEU.mapping, DEU.observable, DEU.input_cov, DEU.parameters)
        dists = (Uniform(1,2), Uniform(3,4), Truncated(Normal(0,1),-5,5))
        gd = GenericDistribution(dists...)
        @testset "DiffEq" begin
            x = [mean(d) for d in dists]
            g(soln,p) = soln[1,end]
            h(x,u,p) = x,p
            prob = ODEProblem(eoms[1], u0s[1], tspan, ps[1])        
            sm = SystemMap(prob, Tsit5(); saveat=1.0)
            ep = @constinferred ExpectationProblem(sm, g, h, gd)
            for foo ∈ getters
                @constinferred foo(ep)
            end
            f = build_integrand(ep, Koopman())
            @constinferred f(x, DEU.parameters(ep))
        end
        @testset "General Map" begin
            f(x,p) = sum(p.*sin.(x))
            ep = @constinferred ExpectationProblem(f, gd, [1.0,1.0,2.0])
            for foo ∈ getters
                @constinferred foo(ep)
            end
            f = build_integrand(ep, Koopman())
            @constinferred f([0.0, 1.0, 2.0], DEU.parameters(ep))
        end
    end
end





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
