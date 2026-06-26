using SciMLTesting, SciMLExpectations, JET, Test

run_qa(
    SciMLExpectations;
    explicit_imports = true,
    aqua_kwargs = (; ambiguities = (; recursive = false)),
    aqua_broken = (:piracies,),  # Base.eltype/minimum/maximum/extrema(::UnivariateKDE), should upstream: https://github.com/SciML/SciMLExpectations.jl/issues/225
    ei_kwargs = (;
        all_qualified_accesses_via_owners = (;
            ignore = (Symbol("@adjoint"),),  # @adjoint owned by ZygoteRules, re-exported by and accessed via Zygote
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                Symbol("@adjoint"),    # Zygote (re-exported from ZygoteRules); still non-public
                :EnsembleAlgorithm,    # SciMLBase; still non-public as of 3.24.0
            ),
        ),
    ),
    ei_broken = (:no_implicit_imports,),  # ~30 names from heavy blanket `using`; explicit-import refactor deferred: https://github.com/SciML/SciMLExpectations.jl/issues/229
)
