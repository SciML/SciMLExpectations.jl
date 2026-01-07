using Test, SciMLExpectations

# JET tests for static analysis
# These tests verify type stability and catch potential runtime errors

@testset "JET static analysis" begin
    using JET

    # Test that the package has no errors when analyzed
    @testset "Package analysis" begin
        rep = JET.report_package(SciMLExpectations; target_modules = (SciMLExpectations,))
        @test length(JET.get_reports(rep)) == 0
    end
end
