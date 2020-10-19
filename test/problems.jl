using DiffEqUncertainty
using OrdinaryDiffEq
using Distributions


function eom!(du,u,p,t)
  @inbounds begin
    du[1] = u[2]
    du[2] = -p[1]*u[1] - p[2]*u[2]
  end
  nothing
end

u0 = [1.0;1.0]
tspan = (0.0,3.0)
p = [1.0, 2.0]
ode_prob = ODEProblem(eom!,u0,tspan,p)
g(sol) = sol[1,end]

# various u0/p distributions
u0_const = [5., 5.]
u0_mixed = [5., Uniform(1.,9.)]
u0_mv_dist = MvNormal([5., 5.], [1., 1.])
p_const = [2., 1f0]
p_mixed = [Uniform(1.,2.), 1f0]

@testset "Expectation problems" begin
    @testset "mixed u0, const p" begin 
        prob = ExpectationProblem(g, u0_mixed, p_const, ode_prob)
        @test prob.Tscalar == Float64
        @test prob.to_quad([1., 2.], [3., 4.]) == ([2.], [1.,3.,4.])
        @test prob.to_phys([2.], [1.,3.,4.]) == ([1., 2.], [3., 4.])
        @test prob.f0_func(u0_const, p_const) == (1. / 8.)
        (x,p) = prob.samp_func()
        @test (x[1] == 5.) && (1. <= x[2] <= 9.) && (p[1] == 2.) && (p[2] == 1.)
        @test (prob.quad_lb, prob.quad_ub) == ([1.], [9.])
        @test prob.p_quad == [5., 2., 1.]
    end

    @testset "mixed u0, mixed p" begin
        prob = ExpectationProblem(g, u0_mixed, p_mixed, ode_prob)
        @test prob.Tscalar == Float64
        @test prob.to_quad([1., 2.], [3., 4.]) == ([2.,3.], [1., 4.])
        @test prob.to_phys([2., 3.], [1., 4.]) == ([1., 2.], [3., 4.])
        @test prob.f0_func(u0_const, p_const) == (1. / 8.)
        (x,p) = prob.samp_func()
        @test (x[1] == 5.) && (1. <= x[2] <= 9.) && (1. <= p[1] <= 2.) && (p[2] == 1.)
        @test (prob.quad_lb, prob.quad_ub) == ([1.,1.,], [9.,2.])
        @test prob.p_quad == [5., 1.]
    end

    @testset "multivariate u0, mixed p" begin
        prob = ExpectationProblem(g, u0_mv_dist, p_mixed, ode_prob)
        @test prob.Tscalar == Float64
        @test prob.to_quad([1., 2.], [3., 4.]) == ([1.,2.,3.], [4.])
        @test prob.to_phys([1.,2.,3.], [4.]) == ([1., 2.], [3., 4.])
        @test prob.f0_func(u0_const, p_const) == pdf(u0_mv_dist, u0_const)
        @test length.(prob.samp_func()) == (2, 2)
        @test (prob.quad_lb, prob.quad_ub) == ([-Inf,-Inf,1.], [Inf, Inf,2.])
        @test prob.p_quad == [1.]
    end

    @testset "multivariate u0, mixed p w/ bounds" begin
        prob = ExpectationProblem(g, u0_mv_dist, p_mixed, ode_prob; 
                lower_bounds=[1.,1.,1.], upper_bounds=[9.,9.,2.])
        @test (prob.quad_lb, prob.quad_ub) == ([1.,1.,1.], [9.,9.,2.])
    end

    @testset "completeion function" begin
        comp_func(x,p) = (x=[2*x[2];x[2]], p=[p[1],2*p[1]])
        prob = ExpectationProblem(g, u0_mixed, p_mixed, ode_prob; comp_func=comp_func)
        @test prob.f0_func(u0_const, p_const) == (1. / 8.)
        (x,p) = prob.samp_func()
        @test (x[1] == 2*x[2]) && (1. <= x[2] <= 9.) && (1. <= p[1] <= 2.) && (p[2] == 2*p[1])
        @test (prob.quad_lb, prob.quad_ub) == ([1.,1.], [9.,2.])
    end

    @testset "no parameters" begin
        F(x,p,t) = -x
        ode_nop = ODEProblem(F, u0, tspan)
        prob = ExpectationProblem(g, u0_mixed, ode_nop)
        @test prob.to_quad([1.,2.],[]) == ([2.], [1.])
        @test prob.to_phys([2.],[1.]) == ([1.,2.],[])
    end
end
