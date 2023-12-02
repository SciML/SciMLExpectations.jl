using SciMLExpectations, Aqua
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
