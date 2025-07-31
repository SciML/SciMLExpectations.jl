using Test, TestExtras

using SciMLExpectations
using Cuba
using StochasticDiffEq
using DiffEqNoiseProcess
using Distributions

f(du, u, p, t) = (du .= u)
g(du, u, p, t) = (du .= u)
u0 = collect(1:4)

W = WienerProcess(0.0, 0.0, 0.0)
prob = SDEProblem(f, g, u0, (0.0, 1.0), noise = W)
sm = ProcessNoiseSystemMap(prob, 8, LambaEM(), abstol = 1e-3, reltol = 1e-3)
cov(x, u, p) = x, p
observed(sol, p) = sol[:, end]
exprob = ExpectationProblem(sm, observed, cov)
sol1 = solve(exprob, Koopman(), ireltol = 1e-3, iabstol = 1e-3, batch = 64,
    quadalg = CubaDivonne())
sol2 = solve(exprob, MonteCarlo(1_000_000))
@testset "Process noise" begin
    @test isapprox(sol1.u, sol2.u, rtol = 1e-2)
end
