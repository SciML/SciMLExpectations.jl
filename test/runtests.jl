if get(ENV, "GROUP", "Core") == "QA"
    using Pkg, TOML

    # Julia 1.10 cannot resolve the nested QA [sources] before SciMLTesting loads.
    qa_dir = joinpath(@__DIR__, "qa")
    qa_project = TOML.parsefile(joinpath(qa_dir, "Project.toml"))
    scimltesting_source = qa_project["sources"]["SciMLTesting"]
    Pkg.activate(qa_dir)
    Pkg.add(
        Pkg.PackageSpec(
            name = "SciMLTesting",
            url = scimltesting_source["url"],
            rev = scimltesting_source["rev"],
        )
    )
    Pkg.develop(Pkg.PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()

    include(joinpath(qa_dir, "qa.jl"))
else
    using SciMLTesting
    run_tests()
end
