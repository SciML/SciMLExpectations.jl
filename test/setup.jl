function pend!(du, u, p, t)
    du[1] = u[2]
    du[2] = -p[1] / p[2] * sin(u[1])
    nothing
end
function pendca!(du, u, p, t)
    du.state.θ = u.state.θ̇
    du.state.θ̇ = -p.g / p.ℓ * sin(u.state.θ)
end
pendsa(u, p, t) = SVector(u[2], -p[1] / p[2] * sin(u[1]))

g(soln) = soln[1, end]
tspan = (0.0, 10.0)

eoms = (pend!, pendsa, pend!, pendca!)
u0s = ([π / 2, 0.0],
    SVector(π / 2, 0.0),
    ComponentArray(state = (θ = π / 2, θ̇ = 0.0)),
    ComponentArray(state = (θ = π / 2, θ̇ = 0.0)))
ps = ([9.807, 1.0],
    SVector(9.807, 1.0),
    ComponentArray(g = 9.807, ℓ = 1.0),
    ComponentArray(g = 9.807, ℓ = 1.0))
