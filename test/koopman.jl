using OrdinaryDiffEq, Distributions,
      DiffEqUncertainty, Test, Quadrature, Cubature, Cuba, 
      FiniteDiff, Zygote, ForwardDiff, DiffEqGPU, DiffEqSensitivity

quadalgs = [HCubatureJL(), CubatureJLh(), CubatureJLp(), CubaSUAVE(), CubaDivonne(), CubaCuhre()]
batchmode = [EnsembleSerial(), EnsembleThreads(), EnsembleCPUArray()]#, EnsembleGPUArray()]

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
prob = ODEProblem(eom!,u0,tspan,p)

A = [0.0 1.0; -p[1] -p[2]]
u0s_dist = [Uniform(1,10), Uniform(2,6)]

@testset "Koopman solve, nout = 1" begin
  @info "Koopman solve, nout = 1"
  g(sol) = sol[1,end]
  exp_prob = ExpectationProblem(g, u0s_dist, p, prob)
  analytical = (exp(A*tspan[end])*mean.(u0s_dist))[1]

  for alg ∈ quadalgs
    @info "$alg"
    @test solve(exp_prob, Koopman(), Tsit5(); quadalg=alg)[1] ≈ analytical rtol=1e-2
  end
end

@testset "Koopman Expectation, nout = 1" begin
  @info "Koopman Expectation, nout = 1"
  g(sol) = sol[1,end]
  analytical = (exp(A*tspan[end])*mean.(u0s_dist))[1]
  
  for alg ∈ quadalgs
    @info "$alg"
    @test expectation(g, prob, u0s_dist, p, Koopman(), Tsit5(); quadalg = alg)[1] ≈ analytical rtol=1e-2
  end
end

@testset "Koopman solve, nout > 1" begin
  @info "Koopman solve, nout > 1"
  g(sol) = sol[:,end]
  exp_prob = ExpectationProblem(g, u0s_dist, p, prob,2)
  analytical = (exp(A*tspan[end])*mean.(u0s_dist))[1]

  for alg ∈ quadalgs
    @info "$alg"
    @test solve(exp_prob, Koopman(), Tsit5(); quadalg=alg)[1] ≈ analytical rtol=1e-2
  end
end

@testset "Koopman Expectation, nout > 1" begin
  @info "Koopman Expectation, nout > 1"
  g(sol) = sol[:,end]
  analytical = (exp(A*tspan[end])*mean.(u0s_dist))
  
  for alg ∈ quadalgs
    @info "$alg"
    @test expectation(g, prob, u0s_dist, p, Koopman(), Tsit5(); quadalg = alg, nout =2) ≈ analytical rtol=1e-2
  end
end

@testset "Koopman solve, nout = 1, batch" begin
  @info  "Koopman solve, nout = 1, batch"
  g(sol) = sol[1,end]
  exp_prob = ExpectationProblem(g, u0s_dist, p, prob)
  analytical = (exp(A*tspan[end])*mean.(u0s_dist))[1]

  for bmode ∈ batchmode
    for alg ∈ quadalgs
      if alg isa HCubatureJL
        continue
      end
      @info "nout = 1, batch mode = $bmode, $alg"
      @test solve(exp_prob, Koopman(), Tsit5(), bmode; quadalg=alg, batch=15)[1] ≈ analytical rtol=1e-2
    end
  end
end

@testset "Koopman Expectation, nout = 1, batch" begin
  @info "Koopman Expectation, nout = 1, batch"
  g(sol) = sol[1,end]
  analytical = (exp(A*tspan[end])*mean.(u0s_dist))[1]

  for bmode ∈ batchmode
    for alg ∈ quadalgs
      if alg isa HCubatureJL
        continue
      end
      @info "nout = 1, batch mode = $bmode, $alg"
      @test expectation(g, prob, u0s_dist, p, Koopman(), Tsit5(), bmode; quadalg = alg, batch = 15)[1] ≈ analytical rtol=1e-2
    end
  end
end

@testset "Koopman solve, nout > 1, batch" begin
  @info "Koopman solve, nout > 1, batch"
  g(sol) = sol[:,end]
  exp_prob = ExpectationProblem(g, u0s_dist, p, prob, 2)
  analytical = (exp(A*tspan[end])*mean.(u0s_dist))

  for bmode ∈ batchmode
    for alg ∈ quadalgs
      if alg isa HCubatureJL
        continue
      end
      @info "nout = 2, batch mode = $bmode, $alg"
      @test solve(exp_prob, Koopman(), Tsit5(), bmode; quadalg=alg, batch=15) ≈ analytical rtol=1e-2
    end
  end
end

@testset "Koopman Expectation, nout > 1, batch" begin
  @info "Koopman Expectation, nout > 1, batch"
  g(sol) = sol[:,end]
  analytical = (exp(A*tspan[end])*mean.(u0s_dist))

  for bmode ∈ batchmode
    for alg ∈ quadalgs
      if alg isa HCubatureJL
        continue
      end
      @info "nout = 2, batch mode = $bmode, $alg"
      res = expectation(g, prob, u0s_dist, p, Koopman(), Tsit5(), bmode; 
                        quadalg = alg, batch = 15, nout=2)
      @test res ≈ analytical rtol=1e-2
    end
  end
end

############## Koopman AD ###############

@testset "Koopman solve AD" begin
  @info "Koopman solve AD"
  g(sol) = sol[1,end]
  loss = function(p, alg;lb=nothing, ub=nothing)
    exp_prob = ExpectationProblem(g, u0s_dist, p, prob;lower_bounds=lb, upper_bounds=ub)
    solve(exp_prob, Koopman(), Tsit5(); quadalg=alg)[1]
  end
  dp1 = FiniteDiff.finite_difference_gradient(p->loss(p, HCubatureJL()),p)
  for alg ∈ quadalgs
    @info "$alg, ForwardDiff"
    alg isa HCubatureJL ? 
      (@test ForwardDiff.gradient(p->loss(p,alg;lb=[1.,2.],ub=[10.,6.]),p) ≈ dp1 rtol=1e-2) :
      (@test_broken ForwardDiff.gradient(p->loss(p,alg),p) ≈ dp1 rtol=1e-2)
    @info "$alg, Zygote"
    @test Zygote.gradient(p->loss(p,alg),p)[1] ≈ dp1 rtol=1e-2
  end
end

@testset "Koopman Expectation AD" begin
  @info "Koopman Expectation AD"
  g(sol) = sol[1,end]
  loss(p, alg) = expectation(g, prob, u0s_dist, p, Koopman(), Tsit5(); quadalg = alg)[1]
  dp1 = FiniteDiff.finite_difference_gradient(p->loss(p, HCubatureJL()),p)
  for alg ∈ quadalgs
    @info "$alg, ForwardDiff"
    @test ForwardDiff.gradient(p->loss(p,alg),p) ≈ dp1 rtol=1e-2
    @info "$alg, Zygote"
    @test Zygote.gradient(p->loss(p,alg),p)[1] ≈ dp1 rtol=1e-2
  end
end

@testset "Koopman solve AD, batch" begin
  @info "Koopman solve AD, batch"
  g(sol) = sol[1,end]
  loss = function(p, alg, bmode;lb=nothing, ub=nothing)
    exp_prob = ExpectationProblem(g, u0s_dist, p, prob;lower_bounds=lb, upper_bounds=ub)
    solve(exp_prob, Koopman(), Tsit5(); quadalg=alg)[1]
  end
  dp1 = FiniteDiff.finite_difference_gradient(p->loss(p, CubatureJLh(), EnsembleThreads()),p)
  for bmode ∈ batchmode
    for alg ∈ quadalgs
      if alg isa HCubatureJL #no batch support
        continue
      end
      @info "$bmode, $alg, ForwardDiff"
      @test_broken ForwardDiff.gradient(p->loss(p,alg,bmode),p) ≈ dp1 rtol=1e-2
      @info "$bmode, $alg, Zygote"
      @test  Zygote.gradient(p->loss(p,alg,bmode),p)[1] ≈ dp1 rtol=1e-2
    end
  end
end

@testset "Koopman Expectation AD, batch" begin
  @info "Koopman Expectation AD, batch"
  g(sol) = sol[1,end]
  loss(p, alg, bmode) = expectation(g, prob, u0s_dist, p, Koopman(), Tsit5(), bmode; quadalg = alg, batch = 10)[1]
  dp1 = FiniteDiff.finite_difference_gradient(p->loss(p, CubatureJLh(), EnsembleThreads()),p)
  for bmode ∈ batchmode
    for alg ∈ quadalgs
      if alg isa HCubatureJL #no batch support
        continue
      end
      @info "$bmode, $alg, ForwardDiff"
      @test ForwardDiff.gradient(p->loss(p,alg,bmode),p) ≈ dp1 rtol=1e-2
      @info "$bmode, $alg, Zygote"
      @test  Zygote.gradient(p->loss(p,alg,bmode),p)[1] ≈ dp1 rtol=1e-2
    end
  end
end

########## Central Moment ##############

function eom!(du,u,p,t)
  @inbounds begin
    du[1] = p[1]*u[1]
  end
  nothing
end

u0 = [1.0]
tspan = (0.0,3.0)
p = [-.3]
prob = ODEProblem(eom!,u0,tspan,p)

u0s_dist = [Uniform(1,10)]

@testset "Koopman Central Moment" begin
  @info "Koopman Central Moment"
  g(sol) = sol[1,end]
  analytical = [0.0, exp(2*p[1]*tspan[end])*var(u0s_dist[1]), 0.0]
  
  for alg ∈ quadalgs
    if alg isa CubaDivonne || alg isa CubaCuhre  #requires 2D spatial integration
      continue
    end
    r = centralmoment(3, g, prob, u0s_dist, p, Koopman(), Tsit5(); 
                      ireltol = 1e-8, iabstol = 1e-8, quadalg = alg) 
    if alg isa CubaSUAVE
      @test_broken r ≈ analytical rtol=1e-2
    else
      @test r ≈ analytical rtol=1e-2
    end
  end
end

@testset "Koopman Central Moment, batch" begin
  @info "Koopman Central Moment, batch"
  g(sol) = sol[1,end]
  analytical = [0.0, exp(2*p[1]*tspan[end])*var(u0s_dist[1]), 0.0]
  
  for bmode ∈ batchmode
    for alg ∈ quadalgs
      if alg isa CubaDivonne || alg isa CubaCuhre || alg isa HCubatureJL #requires 2D spatial integration
        continue
      end
      @info "batch mode = $bmode, alg = $alg"
      r = centralmoment(3, g, prob, u0s_dist, p, Koopman(), Tsit5(), bmode; 
                        ireltol = 1e-8, iabstol = 1e-8, quadalg = alg, batch=15) 
      if alg isa CubaSUAVE
        @test_broken r ≈ analytical rtol=1e-2
      else
        @test r ≈ analytical rtol=1e-2
      end
    end
  end
end
