using OrdinaryDiffEq, DiffEqBase, DiffEqUncertainty, Quadrature, Cubature, Cuba
using DistributionsAD, Distributions#, HCubature
using Zygote, ForwardDiff,DiffEqSensitivity, FiniteDiff

## Continuious System
function fiip(du,u,p,t)
  @inbounds begin
    du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
    du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
  end
  nothing
end

u0 = [1.0;1.0]
p = [1.5,1.0,3.0,1.0];
saveat = 0.1
prob = ODEProblem(fiip,u0,(0.0,10.0),p)
sol = solve(prob,Tsit5())
alg = CubaCuhre()
# alg = HCubatureJL()
cost(sol) = sum(sol)

μs = [6.0,6.0]
u0s_dist = [truncated(Normal(μs[1],2.0),1.0,10.0),truncated(Normal(μs[2],2.0),1.0,10.0)]

function loss_koop(θ, quadalg, args...; kwargs...)
    expectation(cost, prob, u0s_dist, θ, Koopman(), Tsit5(); quadalg=quadalg, ireltol=1e-5, iabstol = 1e-5, saveat=saveat, kwargs...)[1]
end

function loss_mc(θ, args...; kwargs...)
  expectation(cost, prob, u0s_dist, θ, MonteCarlo(), Tsit5(), args...;saveat=saveat, kwargs...)[1]
end

@time loss_koop(p, alg)
@time loss_mc(p; trajectories = 10_000)

@time Zygote.gradient(p->loss_koop(p,alg),p)
ForwardDiff.gradient(p->loss_koop(p,alg),p)
FiniteDiff.finite_difference_gradient(p->loss_koop(p,alg),p)

#### Non-working Example
function loss_koop2(θ, quadalg, args...; kwargs...)
  u0s_dist = [truncated(Normal(θ[1],2.0),1.0,10.0),truncated(Normal(θ[2],2.0),1.0,10.0)]
  expectation(cost, prob, u0s_dist, p, Koopman(), Tsit5(); quadalg=quadalg, ireltol=1e-5, iabstol = 1e-5, saveat=saveat, kwargs...)[1]
end

function loss_mc2(θ, args...; kwargs...)
  u0s_dist = [truncated(Normal(θ[1],2.0),1.0,10.0),truncated(Normal(θ[2],2.0),1.0,10.0)]
  expectation(cost, prob, u0s_dist, p, MonteCarlo(), Tsit5(), args...;saveat=saveat, kwargs...)[1]
end

@time loss_koop2(μs, alg)
@time loss_mc2(μs; trajectories = 10_000)

@time Zygote.gradient(p->loss_koop2(p,alg),μs)
@time ForwardDiff.gradient(p->loss_koop2(p,alg),μs)
FiniteDiff.finite_difference_gradient(p->loss_koop2(p,alg),μs)

##### Working Hyper parameters
function loss_koop3(θ, quadalg, args...; kwargs...)
  u0_f(θ) = [truncated(Normal(θ[1],2.0),1.0,10.0),truncated(Normal(6.0,2.0),1.0,10.0)]
  p_f(θ) = p
  expectation(cost, prob, u0_f, p_f, θ, Koopman(), Tsit5(),args...; quadalg=quadalg, ireltol=1e-5, iabstol = 1e-5, saveat=saveat, kwargs...)[1]
end

@time loss_koop3(μs, alg)
@time Zygote.gradient(p->loss_koop3(p,alg),μs)
@time ForwardDiff.gradient(p->loss_koop3(p,alg),μs)
@time FiniteDiff.finite_difference_gradient(p->loss_koop3(p,alg),μs)

###### Batch Example
batch = 2
@time loss_koop3(μs, CubaCuhre(), EnsembleThreads(); batch = batch)
tp = [6.0,]
@time Zygote.gradient(p->loss_koop3(p,CubaCuhre(), EnsembleThreads(); batch = batch),tp)
@time FiniteDiff.finite_difference_gradient(p->loss_koop3(p,CubaCuhre(), EnsembleThreads(); batch = batch),μs)

# Zygote.gradient(p->loss_koop3(p,CubaCuhre(); batch = batch),p)

# #Discreate Map
# function S(u,p, args...; kwargs...)
#     2.0*u
# end

# function loss2_koop(θ, quadalg, args...; batch=0)
#     expectation(cost, S, u0s_dist, θ, Koopman(), Tsit5(), args...; quadalg=quadalg, ireltol=1e-5, iabstol = 1e-5, saveat=saveat, batch=batch)[1]
# end

# function loss2_mc(θ, args...; kwargs...)
#     expectation(cost, S, u0s_dist, θ, MonteCarlo(), Tsit5(), args...;saveat=saveat, kwargs...)[1]
# end
