# SciMLExpectations.jl: Expectated Values of Simulations and Uncertainty Quantification

SciMLExpectations.jl is a package for quantifying the uncertainties of simulations by
calculating the expectations of observables with respect to input uncertainties. Its goal
is to make it fast and easy to compute solution moments in a differentiable way in order
to enable fast optimization under uncertainty.

## Installation

To install SciMLExpectations.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("SciMLExpectations")
```

## Contributing

- Please refer to the
  [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
  for guidance on PRs, issues, and other matters relating to contributing to SciML.
- There are a few community forums:
    - the #diffeq-bridged channel in the [Julia Slack](https://julialang.org/slack/)
    - [JuliaDiffEq](https://gitter.im/JuliaDiffEq/Lobby) on Gitter
    - on the [Julia Discourse forums](https://discourse.julialang.org)
    - see also [SciML Community page](https://sciml.ai/community/)
