# Reexported integration algorithms

`using SciMLExpectations` also brings the integration algorithm selectors accepted by the
`quadalg` keyword into scope. The names are owned and documented by
[Integrals.jl](https://docs.sciml.ai/Integrals/stable/) and its backends:

  - Built-in and pure-Julia algorithms: `ArblibJL`, `FastTanhSinhQuadratureJL`,
    `GaussLegendre`, `HAdaptiveIntegrationJL`, `HCubatureJL`, `QuadGKJL`,
    `QuadratureRule`, `SimpsonsRule`, and `TrapezoidalRule`
  - Cuba algorithms: `CubaCuhre`, `CubaDivonne`, `CubaSUAVE`, and `CubaVegas`
  - Cubature algorithms: `CubatureJLh` and `CubatureJLp`
  - VEGAS algorithms: `VEGAS` and `VEGASMC`
  - Domain transformation: `ChangeOfVariables`

The corresponding backend package must still be loaded when an algorithm requires one.
Anything else from Integrals must be imported from Integrals directly.
