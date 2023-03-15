using IntegralsCuba
using SciMLExpectations
using StochasticDiffEq
using DiffEqNoiseProcess
using Distributions

f(du, u, p, t) = (du .= u)
g(du, u, p, t) = (du .= u)
u0 = collect(1:4)

W = WienerProcess(0.0, 0.0, 0.0)
prob = SDEProblem(f, g, u0, (0.0, 1.0), noise=W)
sm = ProcessNoiseSystemMap(prob, 5, LambaEM())
cov(x, u, p) = x, p
observed(sol, p) = sol[:, end]
exprob = ExpectationProblem(sm, observed, cov; nout=length(u0))
sol2 = solve(exprob, MonteCarlo(10_000))
sol2.u
sol1 = solve(exprob, Koopman(), ireltol=1e-1, iabstol=1e-1,batch=2,quadalg = CubaSUAVE())
sol1.u
isapprox(sol1.u, sol2.u,rtol=1e-2)
