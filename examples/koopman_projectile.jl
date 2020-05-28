using OrdinaryDiffEq, DiffEqBase, Distributions,
      DiffEqUncertainty, Test, Quadrature, Cubature
# using Plots

include("dirac.jl")

function eom!(du,u,p,t)
    @inbounds begin
        CdS, g, ρ, m = p
        Vb = sqrt(u[3]^2 + u[4]^2)
        coeff = -ρ * CdS * Vb / m #ρ*CdS*V/m
        du[1] = u[3]
        du[2] = u[4]
        du[3] = coeff * u[3]
        du[4] = coeff * u[4] - g
    end
    nothing
end;

cost(x) = abs2(x[1,end] - 25.0)

function ic_func(x)
    [0.0, 0.0, x[1]*cos(x[2]), x[1]*sin(x[2])]
end

quadalg = CubatureJLh()

u0 = ic_func([100.0, 45.0*π/180.0])
ps =[1.0,9.807, 1.277, 10.0]
tspan = (0.,5000.0)

ground_impact(u,t,integrator) = u[2]
affect!(integrator) = terminate!(integrator)
cb = ContinuousCallback(ground_impact,affect!, save_positions=(true,false));

prob = ODEProblem(eom!,u0,tspan,ps)

# how to only return values at callbacks???
sol = solve(prob, Tsit5(),callback = cb);
sol2 = solve(prob, Tsit5(), callback=cb, save_start=false,save_everystep=false,save_end=false)
# plot(sol, vars=(1,2), xlabel="x", ylabel="z", lw=3, leg=nothing, aspect_ratio=:equal)
cost(sol)

## Setup Koopman
δ = 1
u0_params = [Truncated(Normal(100,5),75,125),Uniform((45-δ)*π/180.0,(45+δ)*π/180.0)]
ps  = [Uniform(1-δ,1+δ), Uniform(9.807-δ, 9.807+δ), Uniform(1.277-δ, 1.277+δ), Uniform(10.0,11.0)]

koop_𝔼 = koopman_expectation(cost,u0_params,ps,prob,Tsit5();u0s_func = ic_func, iabstol=1e-3,ireltol=1e-3,maxiters=100000, callback=cb,
                            save_start=false,save_everystep=false,save_end=false, quadalg = quadalg)
c1, e1 = koop_𝔼.u, koop_𝔼.resid

koop_𝔼2 = koopman_expectation2(cost,u0_params,ps,prob,Tsit5();u0s_func = ic_func, iabstol=1e-3,ireltol=1e-3,maxiters=100000, callback=cb,
            save_start=false,save_everystep=false,save_end=false, quadalg = quadalg)
c2, e2 = koop_𝔼2.u, koop_𝔼2.resid

##### Batch
koop_batch_𝔼 = koopman_expectation(cost,u0_params,ps,prob,Tsit5(),EnsembleThreads();u0s_func = ic_func, iabstol=1e-3,ireltol=1e-3,maxiters=100000, callback=cb,
                            save_start=false,save_everystep=false,save_end=false, batch=1000, quadalg = quadalg)
c1_batch, e1_batch = koop_batch_𝔼.u, koop_batch_𝔼.resid

koop_batch_𝔼2 = koopman_expectation2(cost,u0_params,ps,prob,Tsit5(),EnsembleThreads();u0s_func = ic_func, iabstol=1e-3,ireltol=1e-3,maxiters=100000, callback=cb,
                            save_start=false,save_everystep=false,save_end=false, batch=100, quadalg = quadalg)
c1_batch2, e1_batch2 = koop_batch_𝔼2.u, koop_batch_𝔼2.resid

using BenchmarkTools
begin

    u0_params = [Uniform(100-δ,100+δ),Dirac(45*π/180.0)]
    ps  = [Uniform(1-δ,1+δ), Dirac(9.807), Uniform(1.277-δ, 1.277+δ), Uniform(10.0,11.0)]
    @btime koopman_expectation(cost,u0_params,ps,prob,Tsit5();quadalg = quadalg,u0s_func = ic_func, iabstol=1e-3,ireltol=1e-3,maxiters=1000, callback=cb,save_start=false,save_everystep=false,save_end=false,batch=1)
    @btime koopman_expectation2(cost,u0_params,ps,prob,Tsit5();quadalg = quadalg,u0s_func = ic_func, iabstol=1e-3,ireltol=1e-3,maxiters=1000, callback=cb,save_start=false,save_everystep=false,save_end=false, batch=1)

    u0_params = [Uniform(100-δ,100+δ),45*π/180.0]
    ps  = [Uniform(1-δ,1+δ), 9.807, Uniform(1.277-δ, 1.277+δ), Uniform(10.0,11.0)]
    @btime koopman_expectation2(cost,u0_params,ps,prob,Tsit5();quadalg = quadalg, u0s_func = ic_func, iabstol=1e-3,ireltol=1e-3,maxiters=1000, callback=cb,save_start=false,save_everystep=false,save_end=false, batch=1)
end
