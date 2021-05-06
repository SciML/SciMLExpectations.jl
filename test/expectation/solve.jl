using Test, TestExtras,
    DiffEqUncertainty, OrdinaryDiffEq, Distributions,
    StaticArrays, ComponentArrays, Random

const DEU = DiffEqUncertainty
# include("setup.jl")

quadalgs = [HCubatureJL(), CubatureJLh(), CubatureJLp()]#, CubaSUAVE(), CubaDivonne(), CubaCuhre()]
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


@testset "Koopman Expectation, nout = 1" begin
  g(sol) = sol[1,end]
  exprob = ExpectationProblem(sm, g, cov, gd)
  analytical = (exp(A*tspan[end])*[mean(d) for d in u0s_dist])[1]

  for alg ∈ quadalgs
    @info "$alg"
    @test solve(exprob, Koopman(); quadalg = alg)[1] ≈ analytical rtol=1e-2  
    # @test expectation(g, prob, u0s_dist, p, Koopman(), Tsit5(); quadalg = alg)[1] ≈ analytical rtol=1e-2
  end
end