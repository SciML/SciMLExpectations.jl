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

μs = [6.0,6.0]
u0s_dist = [truncated(Normal(μs[1],2.0),1.0,10.0),truncated(Normal(μs[2],2.0),1.0,10.0)]

function loss_koop(θ, quadalg, args...; kwargs...)
    expectation(sum, prob, u0s_dist, θ, Koopman(), Tsit5(); quadalg=quadalg, ireltol=1e-5, iabstol = 1e-5, saveat=saveat, kwargs...)[1]
end

function loss_mc(θ, args...; kwargs...)
  expectation(sum, prob, u0s_dist, θ, MonteCarlo(), Tsit5(), args...;saveat=saveat, kwargs...)[1]
end

@time loss_koop(p, CubaCuhre())
@time loss_mc(p; trajectories = 10_000)

@time Zygote.gradient(p->loss_koop(p,CubaCuhre()),p)
ForwardDiff.gradient(p->loss_koop(p,CubaCuhre()),p)


Zygote.gradient(p->loss_mc(p, trajectories = 10),p)











# @time Zygote.gradient(p->loss2(p,CubaCuhre()),p)

# @run Zygote.gradient(p->loss1(p,CubaCuhre()),p)
# @time FiniteDiff.finite_difference_derivative(p->loss1(p,CubaCuhre()),p)
# @time FiniteDiff.finite_difference_derivative(p->loss2(p,CubaCuhre()),p)

#Discreate Map
function S(u,p, args...; kwargs...)
    2.0*u
end

function loss2_koop(θ, quadalg, args...; batch=0)
    expectation(sum, S, u0s_dist, θ, Koopman(), Tsit5(), args...; quadalg=quadalg, ireltol=1e-5, iabstol = 1e-5, saveat=saveat, batch=batch)[1]
end

function loss2_mc(θ, args...; kwargs...)
    expectation(sum, S, u0s_dist, θ, MonteCarlo(), Tsit5(), args...;saveat=saveat, kwargs...)[1]
end


