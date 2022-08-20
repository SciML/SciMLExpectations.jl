using OrdinaryDiffEq, Distributions,
      SciMLExpectations, Test, Integrals, IntegralsCubature, IntegralsCuba,
      FiniteDiff, Zygote, ForwardDiff, DiffEqGPU, SciMLSensitivity, LinearAlgebra

quadalgs = [
    HCubatureJL(),
    CubatureJLh(),
    CubatureJLp(),
    CubaSUAVE(),
    CubaDivonne(),
    CubaCuhre(),
]
batchmode = [EnsembleSerial(), EnsembleThreads(), EnsembleCPUArray()]#, EnsembleGPUArray()]

function eom!(du, u, p, t)
    @inbounds begin
        du[1] = u[2]
        du[2] = -p[1] * u[1] - p[2] * u[2]
    end
    nothing
end

u0 = [1.0; 1.0]
tspan = (0.0, 3.0)
p = [1.0, 2.0]
prob = ODEProblem(eom!, u0, tspan, p)

A = [0.0 1.0; -p[1] -p[2]]
u0s_dist = [Uniform(1, 10), Uniform(2, 6)]

@testset "Koopman Expectation, nout = 1" begin
    g(sol) = sol[1, end]
    analytical = (exp(A * tspan[end]) * mean.(u0s_dist))[1]

    for alg in quadalgs
        @info "$alg"
        @test expectation(g, prob, u0s_dist, p, Koopman(), Tsit5(); quadalg = alg)[1]≈analytical rtol=1e-2
    end
end

@testset "Koopman Expectation, nout > 1" begin
    g(sol) = sol[:, end]
    analytical = (exp(A * tspan[end]) * mean.(u0s_dist))

    for alg in quadalgs
        @info "$alg"
        @test expectation(g, prob, u0s_dist, p, Koopman(), Tsit5(); quadalg = alg,
                          nout = 2)≈analytical rtol=1e-2
    end
end

# @testset "Koopman Expectation, nout = 1, batch" begin
#   g(sol) = sol[1,end]
#   analytical = (exp(A*tspan[end])*mean.(u0s_dist))[1]

#   for bmode ∈ batchmode
#     for alg ∈ quadalgs
#       if alg isa HCubatureJL
#         continue
#       end
#       @info "nout = 1, batch mode = $bmode, $alg"
#       @test expectation(g, prob, u0s_dist, p, Koopman(), Tsit5(), bmode; quadalg = alg, batch = 15)[1] ≈ analytical rtol=1e-2
#     end
#   end
# end

# @testset "Koopman Expectation, nout > 1, batch" begin
#   g(sol) = sol[:,end]
#   analytical = (exp(A*tspan[end])*mean.(u0s_dist))

#   for bmode ∈ batchmode
#     for alg ∈ quadalgs
#       if alg isa HCubatureJL
#         continue
#       end
#       @info "nout = 2, batch mode = $bmode, $alg"
#       res = expectation(g, prob, u0s_dist, p, Koopman(), Tsit5(), bmode;
#                         quadalg = alg, batch = 15, nout=2)
#       @test res ≈ analytical rtol=1e-2
#     end
#   end
# end

############## Koopman AD ###############

# @testset "Koopman Expectation AD" begin
#   g(sol) = sol[1,end]
#   loss(p, alg) = expectation(g, prob, u0s_dist, p, Koopman(), Tsit5(); quadalg = alg)[1]
#   dp1 = FiniteDiff.finite_difference_gradient(p->loss(p, HCubatureJL()),p)
#   for alg ∈ quadalgs
#     @info "$alg, ForwardDiff"
#     @test ForwardDiff.gradient(p->loss(p,alg),p) ≈ dp1 rtol=1e-2
#     @info "$alg, Zygote"
#     @test Zygote.gradient(p->loss(p,alg),p)[1] ≈ dp1 rtol=1e-2
#   end
# end

# @testset "Koopman Expectation AD, batch" begin
#   g(sol) = sol[1,end]
#   loss(p, alg, bmode) = expectation(g, prob, u0s_dist, p, Koopman(), Tsit5(), bmode; quadalg = alg, batch = 10)[1]
#   dp1 = FiniteDiff.finite_difference_gradient(p->loss(p, CubatureJLh(), EnsembleThreads()),p)
#   for bmode ∈ batchmode
#     for alg ∈ quadalgs
#       if alg isa HCubatureJL #no batch support
#         continue
#       end
#       @info "$bmode, $alg, ForwardDiff"
#       if typeof(alg) <: CubaSUAVE && typeof(bmode) <: EnsembleCPUArray
#         # Passes and fails randomly
#         @test_skip ForwardDiff.gradient(p->loss(p,alg,bmode),p) ≈ dp1 rtol=1e-2
#       else
#         @test ForwardDiff.gradient(p->loss(p,alg,bmode),p) ≈ dp1 rtol=1e-2
#       end
#       @info "$bmode, $alg, Zygote"
#       if typeof(alg) <: CubaCuhre && typeof(bmode) <: EnsembleCPUArray
#         @test_broken  Zygote.gradient(p->loss(p,alg,bmode),p)[1] ≈ dp1 rtol=1e-2
#       elseif typeof(alg) <: Union{CubatureJLh,CubatureJLp,CubaSUAVE,CubaDivonne}
#         # Passes and fails randomly
#         @test_skip  Zygote.gradient(p->loss(p,alg,bmode),p)[1] ≈ dp1 rtol=1e-2
#       else
#         @test  Zygote.gradient(p->loss(p,alg,bmode),p)[1] ≈ dp1 rtol=1e-2
#       end
#     end
#   end
# end

# ########## Central Moment ##############

# function eom!(du,u,p,t)
#   @inbounds begin
#     du[1] = p[1]*u[1]
#   end
#   nothing
# end

# u0 = [1.0]
# tspan = (0.0,3.0)
# p = [-.3]
# prob = ODEProblem(eom!,u0,tspan,p)

# u0s_dist = [Uniform(1,10)]

# @testset "Koopman Central Moment" begin
#   g(sol) = sol[1,end]
#   analytical = [0.0, exp(2*p[1]*tspan[end])*var(u0s_dist[1]), 0.0]

#   for alg ∈ quadalgs
#     if alg isa CubaDivonne || alg isa CubaCuhre  #requires 2D spatial integration
#       continue
#     end
#     r = centralmoment(3, g, prob, u0s_dist, p, Koopman(), Tsit5();
#                       ireltol = 1e-8, iabstol = 1e-8, quadalg = alg)
#     if alg isa CubaSUAVE
#       @test_broken r ≈ analytical rtol=1e-2
#     else
#       @test r ≈ analytical rtol=1e-2
#     end
#   end
# end

# @testset "Koopman Central Moment, batch" begin
#   g(sol) = sol[1,end]
#   analytical = [0.0, exp(2*p[1]*tspan[end])*var(u0s_dist[1]), 0.0]

#   for bmode ∈ batchmode
#     for alg ∈ quadalgs
#       if alg isa CubaDivonne || alg isa CubaCuhre || alg isa HCubatureJL #requires 2D spatial integration
#         continue
#       end
#       @info "batch mode = $bmode, alg = $alg"
#       r = centralmoment(3, g, prob, u0s_dist, p, Koopman(), Tsit5(), bmode;
#                         ireltol = 1e-8, iabstol = 1e-8, quadalg = alg, batch=15)
#       if alg isa CubaSUAVE
#         @test_broken r ≈ analytical rtol=1e-2
#       else
#         @test r ≈ analytical rtol=1e-2
#       end
#     end
#   end
# end

# ########## AbstractODEProblem tests ##############

# function eom!(du,u,p,t)
#   @inbounds begin
#     du[1] = u[2]
#     du[2] = -p[1]*u[1] - p[2]*u[2]
#   end
#   nothing
# end

# u0 = [1.0, 1.0]
# p = [1.0, 2.0]
# prob = ODEForwardSensitivityProblem(eom!,u0,(0.0,3.0),p,saveat=0:3)
# function g(sol)
#     J = extract_local_sensitivities(sol,true)[2]
#     det(J'*J)
# end
# u0_dist = [Uniform(0.7,1.3), 1.0]
# p_dist = [1.0, truncated(Normal(2.0,.1),1.6, 2.4)]
# u0_dist_extended = vcat(u0_dist,zeros(length(p)*length(u0)))

# @testset "ODEForwardSensitivityProblem" begin
#   @test expectation(g, prob, u0_dist_extended, p_dist, MonteCarlo(), Tsit5(); trajectories =100_000) ≈ 0.06781155001419734 rtol=1e-1
#   @test expectation(g, prob, u0_dist_extended, p_dist, Koopman(), Tsit5())[1] ≈ 0.06781155001419734 rtol=1e-1
# end
