using Documenter, DiffEqUncertainty

include("pages.jl")

makedocs(sitename = "DiffEqUncertainty.jl",
         authors = "Chris Rackauckas",
         modules = [DiffEqUncertainty],
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
                                  canonical = "https://DiffEqUncertainty.sciml.ai/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/DiffEqUncertainty.jl.git";
           push_preview = true)
