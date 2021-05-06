using Test, TestExtras,
    DiffEqUncertainty, OrdinaryDiffEq, Distributions,
    Quadrature, Cubature, Cuba,
    StaticArrays, ComponentArrays, Random

const DEU = DiffEqUncertainty
# include("setup.jl")

quadalgs = [HCubatureJL(), CubatureJLh(), CubatureJLp(), CubaSUAVE(), CubaDivonne(), CubaCuhre()]
# batchmode = [EnsembleSerial(), EnsembleThreads()]#, EnsembleGPUArray()]

function eom!(du,u,p,t,A)
  @inbounds begin
    du .= A*u
  end
  nothing
end

u0 = [1.0, 1.0]
tspan = (0.0, 3.0)
p = [1.0; 2.0]
A = [0.0 1.0; -p[1] -p[2]]
prob = ODEProblem((du, u, p, t)->eom!(du, u, p, t, A),u0,tspan,p)
u0s_dist = (Uniform(1,10), Truncated(Normal(3.0,1),0.0,6.0))
gd = GenericDistribution(u0s_dist...)
function cov(x,u,p)
  x,p
end
sm = SystemMap(prob, Tsit5(); save_everystep=false)


@testset "Expectation Correctness" begin
  analytical = (exp(A*tspan[end])*[mean(d) for d in u0s_dist])
  @testset "Scalar Observable (nout = 1)" begin
    g(sol) = sol[1,end]
    exprob = ExpectationProblem(sm, g, cov, gd)
    for alg ∈ quadalgs
      @test solve(exprob, Koopman(); quadalg = alg)[1] ≈ analytical[1] rtol=1e-2
    end
  end
  @testset "Vector-Valued Observable (nout > 1)" begin
    g(sol) = sol[:,end]
    exprob = ExpectationProblem(sm, g, cov, gd; nout = length(u0))
    for alg ∈ quadalgs
      @test solve(exprob, Koopman(); quadalg = alg) ≈ analytical rtol=1e-2
    end
  end
end
