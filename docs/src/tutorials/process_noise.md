# Expectation of Process Noise

SciMLExpectations.jl is able to calculate the average trajectory of a stochastic differential equation.
This done by representing the Wiener process using the [Kosambi–Karhunen–Loève theorem](https://en.wikipedia.org/wiki/Kosambi%E2%80%93Karhunen%E2%80%93Lo%C3%A8ve_theorem#The_Wiener_process).

```@example process_noise
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
sol1.u
```

```@example process_noise
sol2 = solve(exprob, MonteCarlo(1_000_000))
sol2.u
```

In theory, any numerical integration method from Integrals.jl is supported, but in practice many of the techniques struggle with correctly calculating the expected value.
We got the best results with the Divonne algorithm from Cuba.jl.
