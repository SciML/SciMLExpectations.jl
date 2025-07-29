using Test, TestExtras,
      SciMLExpectations, OrdinaryDiffEq, Distributions,
      StaticArrays, ComponentArrays, Random, ForwardDiff

const DEU = SciMLExpectations
include("setup.jl")

@testset "GenericDistribution" begin
    dists = (Uniform(1, 2), Uniform(3, 4), Normal(0, 1), truncated(Normal(0, 1), -3, 3))
    x = [mean(d) for d in dists]

    pdf_f = let dists = dists
        x -> mapreduce(pdf, *, dists, x)
    end
    rand_f = let dists = dists
        () -> [rand(d) for d in dists]
    end
    lb = [minimum.(dists)...]
    ub = [maximum.(dists)...]

    P = product_distribution([dists...])
    gd_ind = @constinferred GenericDistribution(dists...)
    gd_raw = @constinferred GenericDistribution(pdf_f, rand_f, lb, ub)
    @constinferred GenericDistribution(pdf_f, rand_f, lb, ub)

    for gd in (gd_ind, gd_raw)
        @test minimum(gd) == minimum(P)
        @test maximum(gd) == maximum(P)
        @test extrema(gd) == extrema(P)
        @test pdf(gd, x) ≈ pdf(P, x)
        @constinferred pdf(gd, x)

        Random.seed!(0)
        @test rand(gd) == begin
            Random.seed!(0)
            rand(P)
        end
        @constinferred rand(gd)
    end
end

@testset "SystemMap" begin
    for (f, u, p) in zip(eoms, u0s, ps)
        prob = ODEProblem(f, u, tspan, p)
        sm = @constinferred SystemMap(prob, Tsit5(); saveat = 1.0)
        sm_soln = @constinferred sm(u, p)
        soln = solve(prob, Tsit5(); saveat = 1.0)
        @test sm_soln.t == soln.t
        @test sm_soln.u == soln.u
    end
end

@testset "ExpectationProblem" begin
    @testset "Interface" begin
        getters = (DEU.distribution, DEU.mapping, DEU.observable, DEU.input_cov,
            DEU.parameters)
        dists = (Uniform(1, 2), Uniform(3, 4), truncated(Normal(0, 1), -5, 5))
        gd = GenericDistribution(dists...)
        x = [mean(d) for d in dists]
        @testset "DiffEq" begin
            h(x, u, p) = x, p
            prob = ODEProblem(eoms[1], u0s[1], tspan, ps[1])
            sm = SystemMap(prob, Tsit5(); saveat = 1.0)

            # nout = 1
            g(soln, p) = soln[1, end]
            ep = @constinferred ExpectationProblem(sm, g, h, gd)
            for foo in getters
                @constinferred foo(ep)
            end
            f = @constinferred build_integrand(
                ep, Koopman(), x, DEU.parameters(ep), nothing)
            @constinferred f(x, DEU.parameters(ep))

            fbatch = build_integrand(ep, Koopman(), x, DEU.parameters(ep), 10) #= @constinferred =#
            y = reshape(repeat(x, outer = 5), :, 5)
            dy = similar(y[1, :])
            @constinferred fbatch(dy, y, DEU.parameters(ep))

            # nout > 1
            g2(soln, p) = [soln[1, end], soln[2, end]]
            ep = @constinferred ExpectationProblem(sm, g2, h, gd)
            f = @constinferred build_integrand(
                ep, Koopman(), x, DEU.parameters(ep), nothing)
            @constinferred f(x, DEU.parameters(ep))

            fbatch = build_integrand(ep, Koopman(), x, DEU.parameters(ep), 10) #= @constinferred =#
            y = reshape(repeat(x, outer = 5), :, 5)
            dy = similar(y[1:2, :])
            @constinferred fbatch(dy, y, DEU.parameters(ep))
        end
        @testset "General Map" begin
            f(x, p) = sum(p .* sin.(x))
            ep = @constinferred ExpectationProblem(f, gd, [1.0, 1.0, 2.0])
            for foo in getters
                @constinferred foo(ep)
            end
            f = @constinferred build_integrand(
                ep, Koopman(), x, DEU.parameters(ep), nothing)
            @constinferred f([0.0, 1.0, 2.0], DEU.parameters(ep))
        end
    end
end
