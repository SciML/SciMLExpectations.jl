using SciMLTesting, SciMLExpectations, JET, Test

integrals_reexports = (
    :ArblibJL,
    :BatchIntegralFunction,
    :ChangeOfVariables,
    :CubaCuhre,
    :CubaDivonne,
    :CubaSUAVE,
    :CubaVegas,
    :CubatureJLh,
    :CubatureJLp,
    :FastTanhSinhQuadratureJL,
    :GaussLegendre,
    :HAdaptiveIntegrationJL,
    :HCubatureJL,
    :IntegralFunction,
    :IntegralProblem,
    :Integrals,
    :QuadGKJL,
    :QuadratureRule,
    :ReturnCode,
    :SampledIntegralProblem,
    :SciMLBase,
    :SimpsonsRule,
    :TrapezoidalRule,
    :VEGAS,
    :VEGASMC,
    :init,
    :isinplace,
    :remake,
    :solve,
    :solve!,
    :transformation_cot_inf,
    :transformation_if_inf,
    :transformation_tan_inf,
)

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
            ),
        ),
    ),
    ei_broken = (:no_implicit_imports,),  # ~30 names from heavy blanket `using`; explicit-import refactor deferred: https://github.com/SciML/SciMLExpectations.jl/issues/229
    api_docs_kwargs = (;
        rendered = true,
        # Re-exported from Integrals.jl; docs remain owned by the defining packages.
        rendered_ignore = integrals_reexports,
    ),
)
