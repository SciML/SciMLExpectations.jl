using SciMLExpectations, OrdinaryDiffEq, Distributions, BenchmarkTools
using StableRNGs

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# Pendulum ODE
function pend!(du, u, p, t)
    du[1] = u[2]
    du[2] = -9.807 * sin(u[1])
    return nothing
end
u0 = [0.0, 1.0]
tspan = (0.0, 10.0)
prob = ODEProblem(pend!, u0, tspan)

sm = SystemMap(prob, Tsit5(); saveat = 1.0)
h(x, u, p) = (u, p)
g(soln, p) = soln
gd = product_distribution(Uniform(0.9, 1.1), Uniform(0.9, 1.1))
ep = ExpectationProblem(sm, g, h, gd)

# =============================================================================
# Expectation solves
# =============================================================================

SUITE["expectation"] = BenchmarkGroup()

SUITE["expectation"]["koopman"] = @benchmarkable solve($ep, Koopman())
SUITE["expectation"]["montecarlo"] = @benchmarkable solve($ep, MonteCarlo(500))

# =============================================================================
# Construction
# =============================================================================

SUITE["construct"] = BenchmarkGroup()

SUITE["construct"]["systemmap"] = @benchmarkable SystemMap(
    $prob, Tsit5(); saveat = 1.0
)
SUITE["construct"]["expectation_problem"] = @benchmarkable ExpectationProblem(
    $sm, $g, $h, $gd
)
