"""
`ExpectationSolution`

The solution to an `ExpectationProblem`

## Fields

  - `u`
  - `resid`
  - `original`
"""
struct ExpectationSolution{uType, R, O}
    u::uType
    resid::R
    original::O
end
