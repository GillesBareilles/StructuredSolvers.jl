using LinearAlgebra


"""
    simplify_ordinates(coords)

Simplify a piece-wise constant signal by removing interior of constant areas.
"""
function simplify_ordinates(absciss, ordinates)
    @assert length(absciss) == length(ordinates)

    new_absciss = [ absciss[1] ]
    new_ordinates = [ ordinates[1] ]


    for coord_ind in 2:length(absciss)-1
        if !(ordinates[coord_ind] == ordinates[coord_ind-1] && ordinates[coord_ind] == ordinates[coord_ind+1])
            push!(new_absciss, absciss[coord_ind])
            push!(new_ordinates, ordinates[coord_ind])
        end
    end
    push!(new_absciss, absciss[end])
    push!(new_ordinates, ordinates[end])

    return new_absciss, new_ordinates
end

dim_identified_structure(M::l1Manifold) = sum(1 .- M.nnz_coords)

function get_manifold_distance(M::l1Manifold, M_ref::l1Manifold)
    return dot(1 .- M.nnz_coords, 1 .- M_ref.nnz_coords)
end

function get_manifold_distance(Ms::AbstractVector, M_ref::l1Manifold)
    return [ get_manifold_distance(M, M_ref) for M in Ms]
end

function run_expenums(class_to_problems, algorithms; FIGS_FOLDER="./figs")

    #
    ### Execute all algorithms
    #
    pbclass_to_pbsalgos = OrderedDict()
    # For each problem class
    for (pbclass_name, problems) in class_to_problems
        printstyled("\n\n---- Class $(pbclass_name)\n", color=:light_blue)

        algo_to_pbstraces = Dict()
        for (algo) in algorithms
            printstyled("---- Algorithm $(algo.name)\n", color=:light_blue)

            pb_to_traces = Dict()
            for pb in problems
                printstyled("\n----- Solving problem...\n", color=:light_blue)

                pb_to_traces[pb] = optimize!(pb.pb, algo.optimizer, pb.xstart; algo.params...)
            end
            algo_to_pbstraces[algo] = pb_to_traces
        end
        pbclass_to_pbsalgos[pbclass_name] = algo_to_pbstraces
    end

    #
    ### Produce plots
    #
    for (pbclass_name, algo_to_pbstraces) in pbclass_to_pbsalgos
        plotdata = []

        for (algo, pb_to_traces) in algo_to_pbstraces
            for (pb_id, pbtrace) in enumerate(pb_to_traces)
                pb, trace = pbtrace

                # Produce indicator from iterates
                # indicator is number of elementary manifolds shared by iterate and ground truth signal.
                M_x0 = pb.pb.M_x0
                Ms = [state.additionalinfo.M for state in trace]

                abscisses = [state.it for state in trace]
                ordinates = get_manifold_distance(Ms, M_x0)
                abscisses, ordinates = simplify_ordinates(abscisses, ordinates)

                # normalize coordinates
                ordinates = ordinates .* 100 / dim_identified_structure(M_x0)

                plotoptions = StructuredSolvers.get_curve_params(algo.optimizer, 1, 1, 1)
                pb_id != 1 && (plotoptions["forget plot"] = nothing)

                push!(
                    plotdata,
                    PlotInc(
                        PGFPlotsX.Options(plotoptions...),
                        Coordinates(abscisses, ordinates),
                    ),
                )
                pb_id==1 && push!(plotdata, LegendEntry(StructuredSolvers.get_legendname(algo.optimizer)))
            end
        end

        fig = @pgf Axis(
            {
                # xlabel = "iter",
                # ylabel = L"F(x_k)-F^\star",
                # ymode  = "log",
                xmode  = "log",
                # title = "Problem $(pb.name) -- Suboptimality",
                legend_pos = "outer north east",
                legend_cell_align = "left",
                legend_style = "font=\\footnotesize",
            },
            plotdata...
        )

        try
            pgfsave(joinpath(FIGS_FOLDER, "reg_$(pbclass_name)_identified.pdf"), fig)
        catch e
            println("Error while building pdf:")
            println(e)
        end
        pgfsave(joinpath(FIGS_FOLDER, "reg_$(pbclass_name)_identified.tikz"), fig)
    end
    return
end





function runexpnums_(problems, algorithms; FIGS_FOLDER = "./figs/expnums")

    # ## Start from interesting point and optimize with ISTA, FISTA.
    # problemclass_to_algstats = OrderedDict()
    # for (problems_class_name, problem_class) in problems
    #     printstyled("\n---- Class $(problems_class_name)\n", color=:light_blue)

    #     algo_to_stats = OrderedDict()
    #     for (algo_id, algo) in enumerate(algorithms)
    #         printstyled("\n---- Algorithm $(algo.name)\n", color=:light_blue)

    #         algo_to_stats[algo] = Dict()
    #         for (pb_id, pb_data) in enumerate(problem_class)
    #             printstyled("\n----- Solving problem $pb_id, $(pb_data.name)\n", color=:light_blue)

    #             xopt, hist = solve_proxgrad(pb_data.pb, pb_data.xstart, algo.updatefunc; algo.params...)
    #             algo_to_stats[algo][pb_data] = (hist=hist, xopt=xopt)
    #         end
    #     end

    #     problemclass_to_algstats[problems_class_name] = algo_to_stats
    # end


    ####################################################
    ## Display suboptimality, iterates position.
    basename(pwd()) == "src" && (FIGS_FOLDER = joinpath("..", FIGS_FOLDER))

    for (pbclass_name, algo_to_stats) in problemclass_to_algstats
        println("Generating graphs for class ", pbclass_name)

        ps = []

        for (algo, stats) in algo_to_stats
            @show algo.name

            ############################################
            ## Plot one for legend
            (pb, pbinstance_stats) = first(stats); delete!(stats, pb)

            reg = typeof(pb.pb).parameters[1]
            opt_supp = get_support(reg, pb.pb.x0)

            coords_nbman_y_identified = simplify_coords([ (k, length(intersect(v, opt_supp))) for (k, v) in pbinstance_stats.hist[:iter_support_y]])
            coords_nbman_y_identified = normalize_identified_coords(coords_nbman_y_identified, opt_supp)

            push!(ps, PlotInc(
                PGFPlotsX.Options(
                    "no marks" => nothing,
                    get_iterates_algoparams(algo.name)...
                ),
                Coordinates(coords_nbman_y_identified)
            ))

            for (pb, pbinstance_stats) in stats
                reg = typeof(pb.pb).parameters[1]
                opt_supp = get_support(reg, pb.pb.x0)

                coords_nbman_y_identified = simplify_coords([ (k, length(intersect(v, opt_supp))) for (k, v) in pbinstance_stats.hist[:iter_support_y]])
                coords_nbman_y_identified = normalize_identified_coords(coords_nbman_y_identified, opt_supp)

                push!(ps, PlotInc(
                    PGFPlotsX.Options(
                        "no marks" => nothing,
                        "forget plot" => nothing,
                        get_iterates_algoparams(algo.name)...
                    ),
                    Coordinates(coords_nbman_y_identified)
                ))

            end
            push!(ps, LegendEntry(get_legend(algo.name)))

        end

        fig = @pgf Axis(
            {
                # height = "12cm",
                # width = "12cm",
                # xlabel = "iter",
                # ylabel = L"F(x_k)-F^\star",
                # ymode  = "log",
                xmode  = "log",
                # title = "Problem $(pb.name) -- Suboptimality",
                legend_pos = "north west",
                legend_cell_align = "left",
                legend_style = "font=\\footnotesize",
            },
            ps...
        )
        try
            pgfsave(joinpath(FIGS_FOLDER, "reg_$(pbclass_name)_identified.pdf"), fig)
        catch e
            println("Error while building pdf:")
            println(e)
        end
        pgfsave(joinpath(FIGS_FOLDER, "reg_$(pbclass_name)_identified.tikz"), fig)
        ##################################
    end

    return
end
