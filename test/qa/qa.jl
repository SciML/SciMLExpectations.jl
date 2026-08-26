using SciMLTesting, SciMLExpectations, Test

const REEXPORTS = (
    :ArblibJL, :ChangeOfVariables, :CubaCuhre, :CubaDivonne, :CubaSUAVE, :CubaVegas,
    :CubatureJLh, :CubatureJLp, :FastTanhSinhQuadratureJL, :GaussLegendre,
    :HAdaptiveIntegrationJL, :HCubatureJL, :QuadGKJL, :QuadratureRule, :SimpsonsRule,
    :TrapezoidalRule, :VEGAS, :VEGASMC,
)

run_qa(SciMLExpectations; reexports_allow = REEXPORTS)

@testset "Reexport surface" begin
    @testset "$name" for name in REEXPORTS
        @test name in names(SciMLExpectations)
        @test isdefined(@__MODULE__, name)
    end
end
