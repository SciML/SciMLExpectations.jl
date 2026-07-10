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
            ),
        ),
    ),
    ei_broken = (:no_implicit_imports,),  # ~30 names from heavy blanket `using`; explicit-import refactor deferred: https://github.com/SciML/SciMLExpectations.jl/issues/229
)

function _owned_public_names(mod::Module)
    public_names = Symbol[]
    for name in names(mod; all = false, imported = true)
        name === nameof(mod) && continue
        isdefined(mod, name) || continue
        value = getproperty(mod, name)
        applicable(parentmodule, value) || continue
        parentmodule(value) === mod || continue
        push!(public_names, name)
    end
    return sort!(unique(public_names))
end

function _docstring_text(mod::Module, name::Symbol)
    md = Docs.doc(Docs.Binding(mod, name))
    return sprint(show, MIME"text/plain"(), md)
end

function _manual_docs_entries(mod::Module)
    docs_src = joinpath(pkgdir(mod), "docs", "src")
    entries = String[]
    for (root, _, files) in walkdir(docs_src)
        for file in files
            endswith(file, ".md") || continue
            in_docs_block = false
            for line in eachline(joinpath(root, file))
                stripped = strip(line)
                if startswith(stripped, "```@docs") || startswith(stripped, "```@autodocs")
                    in_docs_block = true
                elseif startswith(stripped, "```")
                    in_docs_block = false
                elseif in_docs_block && !isempty(stripped) && !startswith(stripped, "#")
                    push!(entries, stripped)
                end
            end
        end
    end
    return entries
end

function _entry_documents_name(entry::String, name::Symbol)
    name_string = String(name)
    qualified_name = string(nameof(SciMLExpectations), ".", name_string)
    return entry == name_string ||
        startswith(entry, string(name_string, "(")) ||
        entry == qualified_name ||
        startswith(entry, string(qualified_name, "("))
end

@testset "public API documentation" begin
    manual_docs_entries = _manual_docs_entries(SciMLExpectations)

    for name in _owned_public_names(SciMLExpectations)
        @test !occursin("No documentation found", _docstring_text(SciMLExpectations, name))
        @test any(entry -> _entry_documents_name(entry, name), manual_docs_entries)
    end
end
