include("run_solvers.jl")

function plot_optimdata(optimdata)
    fig = TikzDocument()

    push!(fig, TikzPicture(StructuredSolvers.plot_fvals_iteration(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_tangentres_iteration(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_normalres_iteration(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_structure_iteration(optimdata, M_opt)))
    push!(fig, TikzPicture(StructuredSolvers.plot_step_iteration(optimdata)))

    push!(fig, TikzPicture(StructuredSolvers.plot_fvals_time(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_tangentres_time(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_normalres_time(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_structure_time(optimdata, M_opt)))
    push!(fig, TikzPicture(StructuredSolvers.plot_step_time(optimdata)))

    println("Building output pdf $(pbname) ...")
    PGFPlotsX.pgfsave("figs/$(pbname).pdf", fig)
    PGFPlotsX.pgfsave("figs/$(pbname).tex", fig)
end


function main()

    problems = [
        (   pbname = "lasso-test",
            pb = get_lasso_MLE(n=100, m=130, sparsity=0.5),
            x0 = zeros(n),

        )
    ]


    #
    ### Running algorithms
    #
    printstyled("-- running solvers...\n", color=:red)
    run_solvers!(optimdata)

    #
    ### Plotting data
    #
    plot_optimdata(optimdata)


    #
    ### Building output table
    #

end