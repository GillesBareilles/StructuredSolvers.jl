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


"""
    nb_identified_manifolds(M::Manifold)

Counts the number of "identified manifolds", monotone with manifold codimension. For l1,
number of zeros; for fixed rank, `min(m, n)-k`...
"""
nb_identified_manifolds(M::Manifold) = 0
nb_identified_manifolds(M::l1Manifold) = sum(1 .- M.nnz_coords)
nb_identified_manifolds(M::Euclidean) = 0
nb_identified_manifolds(M::PSphere) = 1
nb_identified_manifolds(M::ProductManifold) = sum(nb_identified_manifolds(man) for man in M.manifolds)
nb_identified_manifolds(::FixedRankMatrices{m, n, k}) where {m, n, k} = min(m, n)-k

"""
nb_correctlyidentified_manifolds(M::Manifold, M_ref::Manifold)

Counts the number of "identified manifolds" of `M` present in `M_ref`.
"""
nb_correctlyidentified_manifolds(M::Manifold, M_ref::Manifold) = 0
nb_correctlyidentified_manifolds(M::l1Manifold, M_ref::l1Manifold) = dot(1 .- M.nnz_coords, 1 .- M_ref.nnz_coords)
nb_correctlyidentified_manifolds(M::Euclidean, M_ref::Euclidean) = 0
nb_correctlyidentified_manifolds(M::PSphere, M_ref::PSphere) = M == M_ref
nb_correctlyidentified_manifolds(M::ProductManifold, M_ref::ProductManifold) = sum(nb_correctlyidentified_manifolds(M.manifolds[i], M_ref.manifolds[i]) for i in 1:length(M.manifolds))
nb_correctlyidentified_manifolds(::FixedRankMatrices{m, n, k}, ::FixedRankMatrices{m, n, k_ref}) where {m, n, k, k_ref} = min(min(m, n)-k, min(m, n)-k_ref)


function get_proportion_identifiedstructure(Ms::AbstractVector, M_ref)
    return [ nb_correctlyidentified_manifolds(M, M_ref) for M in Ms] ./ nb_identified_manifolds(M_ref)
end

function get_identification_ind(Ms, M_x0)
    ind_id = length(Ms)
    while Ms[ind_id] == M_x0
        ind_id -= 1
    end
    has_identified = ind_id != length(Ms)
    return has_identified, ind_id+1
end

function get_finalmanifold_ind(Ms)
    ind_id = length(Ms)
    @show length(Ms)
    while Ms[ind_id] == Ms[end]
        ind_id -= 1
    end
    @show ind_id
    return ind_id+1
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
                ordinates = get_proportion_identifiedstructure(Ms, M_x0)
                abscisses_light, ordinates_light = simplify_ordinates(abscisses, ordinates)

                ## remove first null section
                if first(ordinates_light) == 0
                    abscisses_light, ordinates_light = abscisses_light[2:end], ordinates_light[2:end]
                end

                plotoptions = StructuredSolvers.get_curve_params(algo.optimizer, 1, 1, 1)
                pb_id != 1 && (plotoptions["forget plot"] = nothing)

                push!(
                    plotdata,
                    PlotInc(
                        PGFPlotsX.Options(plotoptions...),
                        Coordinates(abscisses_light, ordinates_light),
                    ),
                )
                pb_id==1 && push!(plotdata, LegendEntry(StructuredSolvers.get_legendname(algo.optimizer)))

                # add identification time
                has_identified, ind_id = get_identification_ind(Ms, M_x0)
                if has_identified
                    push!(
                        plotdata,
                        PlotInc(
                            PGFPlotsX.Options("forget plot"=>nothing, "draw"=>"none", "mark"=>"oplus", "color"=>plotoptions["color"]),
                            Coordinates([abscisses[ind_id]], [ordinates[ind_id]]),
                        ),
                    )
                end

                # ind_id = get_finalmanifold_ind(Ms)
                # push!(
                #     plotdata,
                #     PlotInc(
                #         PGFPlotsX.Options("forget plot"=>nothing, "draw"=>"none", "mark"=>"triangle", "color"=>plotoptions["color"]),
                #         Coordinates([abscisses[ind_id]], [ordinates[ind_id]]),
                #     ),
                # )
            end
        end

        fig = @pgf Axis(
            {
                # xlabel = "iter",
                # ylabel = L"F(x_k)-F^\star",
                # ymode  = "log",
                # xmode  = "log",
                # title = "Problem $(pb.name) -- Suboptimality",
                legend_pos = "outer north east",
                legend_cell_align = "left",
                legend_style = "font=\\footnotesize",
            },
            plotdata...
        )

        try
            pgfsave(joinpath(FIGS_FOLDER, "reg_$(pbclass_name)_identified.pdf"), fig)
            println("Output files: ", joinpath(FIGS_FOLDER, "reg_$(pbclass_name)_identified.pdf"))
        catch e
            println("Error while building pdf:")
            println(e)
        end
        pgfsave(joinpath(FIGS_FOLDER, "reg_$(pbclass_name)_identified.tikz"), fig)
        println("Output files: ", joinpath(FIGS_FOLDER, "reg_$(pbclass_name)_identified.tikz"))
    end
    return
end




function run_expenums_moyenne(class_to_problems, algorithms; FIGS_FOLDER="./figs")

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

            pb_to_ordiante = []

            ## Collect performance indicator through all runs
            for (pb_id, pbtrace) in enumerate(pb_to_traces)
                pb, trace = pbtrace

                # Produce indicator from iterates
                # indicator is number of elementary manifolds shared by iterate and ground truth signal.
                M_x0 = pb.pb.M_x0
                Ms = [state.additionalinfo.M for state in trace]

                abscisses = [state.it for state in trace]
                ordinates = get_proportion_identifiedstructure(Ms, M_x0)
                # abscisses_light, ordinates_light = simplify_ordinates(abscisses, ordinates)

                # ## remove first null section
                # if first(ordinates_light) == 0
                #     abscisses_light, ordinates_light = abscisses_light[2:end], ordinates_light[2:end]
                # end
                push!(pb_to_ordiante, ordinates)
            end

            ordiantes_average = Vector([ mean([pb_to_ordiante[i][j] for i in 1:length(pb_to_ordiante)]) for j in 1:length(first(pb_to_ordiante)) ])

            plotoptions = StructuredSolvers.get_curve_params(algo.optimizer, 1, 1, 1)

            pb_id = 1
            # pb_id != 1 && (plotoptions["forget plot"] = nothing)

            push!(
                plotdata,
                PlotInc(
                    PGFPlotsX.Options(plotoptions...),
                    Coordinates([Float64(i) for i in 1:length(first(pb_to_ordiante))], ordiantes_average),
                ),
            )
            pb_id==1 && push!(plotdata, LegendEntry(StructuredSolvers.get_legendname(algo.optimizer)))

                # # add identification time
                # has_identified, ind_id = get_identification_ind(Ms, M_x0)
                # if has_identified
                #     push!(
                #         plotdata,
                #         PlotInc(
                #             PGFPlotsX.Options("forget plot"=>nothing, "draw"=>"none", "mark"=>"oplus", "color"=>plotoptions["color"]),
                #             Coordinates([abscisses[ind_id]], [ordinates[ind_id]]),
                #         ),
                #     )
                # end

                # ind_id = get_finalmanifold_ind(Ms)
                # push!(
                #     plotdata,
                #     PlotInc(
                #         PGFPlotsX.Options("forget plot"=>nothing, "draw"=>"none", "mark"=>"triangle", "color"=>plotoptions["color"]),
                #         Coordinates([abscisses[ind_id]], [ordinates[ind_id]]),
                #     ),
                # )
            # end
        end

        fig = @pgf Axis(
            {
                # xlabel = "iter",
                # ylabel = L"F(x_k)-F^\star",
                # ymode  = "log",
                # xmode  = "log",
                # title = "Problem $(pb.name) -- Suboptimality",
                legend_pos = "outer north east",
                legend_cell_align = "left",
                legend_style = "font=\\footnotesize",
            },
            plotdata...
        )

        try
            pgfsave(joinpath(FIGS_FOLDER, "reg_$(pbclass_name)_identified.pdf"), fig)
            println("Output files: ", joinpath(FIGS_FOLDER, "reg_$(pbclass_name)_identified.pdf"))
        catch e
            println("Error while building pdf:")
            println(e)
        end
        pgfsave(joinpath(FIGS_FOLDER, "reg_$(pbclass_name)_identified.tikz"), fig)
        println("Output files: ", joinpath(FIGS_FOLDER, "reg_$(pbclass_name)_identified.tikz"))
    end
    return
end
