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

function Base.show(io::IO, ::MIME"text/plain", sol::ExpectationSolution)
    println(io, "ExpectationSolution")
    println(io, "────────────────────")
    
    # Show expectation value(s)
    print(io, "Expectation: ")
    if isa(sol.u, AbstractArray) && length(sol.u) > 5
        # For large arrays, show first few elements with ellipsis
        print(io, "[")
        for i in 1:min(3, length(sol.u))
            print(io, sol.u[i])
            i < min(3, length(sol.u)) && print(io, ", ")
        end
        print(io, ", ..., ", sol.u[end], "]")
        println(io, " (", length(sol.u), " elements)")
    else
        println(io, sol.u)
    end
    
    # Show residual if available
    if !isnothing(sol.resid)
        print(io, "Residual:    ")
        if isa(sol.resid, AbstractArray) && length(sol.resid) > 5
            print(io, "[")
            for i in 1:min(3, length(sol.resid))
                print(io, sol.resid[i])
                i < min(3, length(sol.resid)) && print(io, ", ")
            end
            print(io, ", ..., ", sol.resid[end], "]")
            println(io, " (", length(sol.resid), " elements)")
        else
            println(io, sol.resid)
        end
    end
end

function Base.show(io::IO, sol::ExpectationSolution)
    if isa(sol.u, AbstractArray) && length(sol.u) > 1
        print(io, "ExpectationSolution(u=", length(sol.u), "-element ", typeof(sol.u), ")")
    else
        print(io, "ExpectationSolution(u=", sol.u, ")")
    end
end
