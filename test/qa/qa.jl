using SciMLExpectations, Aqua, JET, Test

@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(SciMLExpectations)
    Aqua.test_ambiguities(SciMLExpectations, recursive = false)
    Aqua.test_deps_compat(SciMLExpectations)
    Aqua.test_piracies(SciMLExpectations, broken = true)
    Aqua.test_project_extras(SciMLExpectations)
    Aqua.test_stale_deps(SciMLExpectations)
    Aqua.test_unbound_args(SciMLExpectations)
    Aqua.test_undefined_exports(SciMLExpectations)
end

@testset "JET" begin
    @test_broken false  # JET: no matching method logpdf(::Int64, ::Int64) in GenericDistribution pdf_func — see https://github.com/SciML/SciMLExpectations.jl/issues/225
end
