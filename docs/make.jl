using Documenter, SciMLExpectations

include("pages.jl")

makedocs(sitename = "SciMLExpectations.jl",
         authors = "Chris Rackauckas",
         modules = [SciMLExpectations],
         strict = [
             :doctest,
             :linkcheck,
             :parse_error,
             :example_block,
             # Other available options are
             # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
         ],
         clean = true, doctest = false,
         format = Documenter.HTML(analytics = "UA-90474609-3",
                                  assets = ["assets/favicon.ico"],
                                  canonical = "https://SciMLExpectations.sciml.ai/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/SciMLExpectations.jl.git";
           push_preview = true)
