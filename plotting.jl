"""
    simplify_ordinates(coords)

Simplify a piece-wise constant signal by removing interior of constant areas.
"""
function simplify_ordinates(absciss, ordinates)
    @assert length(absciss) == length(ordinates)

    new_absciss = [absciss[1]]
    new_ordinates = [ordinates[1]]


    for coord_ind in 2:(length(absciss) - 1)
        if !(
            ordinates[coord_ind] == ordinates[coord_ind - 1] &&
            ordinates[coord_ind] == ordinates[coord_ind + 1]
        )
            push!(new_absciss, absciss[coord_ind])
            push!(new_ordinates, ordinates[coord_ind])
        end
    end
    push!(new_absciss, absciss[end])
    push!(new_ordinates, ordinates[end])

    return new_absciss, new_ordinates
end

function save_fig(fig, figname, FIGS_FOLDER)
    try
        pgfsave(joinpath(FIGS_FOLDER, "$(figname).pdf"), fig)
        println("Output files: ", joinpath(FIGS_FOLDER, "$(figname).pdf"))
    catch e
        println("Error while building pdf:")
        println(e)
    end
    pgfsave(joinpath(FIGS_FOLDER, "$(figname).tikz"), fig)
    return println("Output files: ", joinpath(FIGS_FOLDER, "$(figname).tikz"))
end



"""
    plot_all_curves(algo_pb_curve, figname; FIGS_FOLDER = ".")

Expects a collection `algo_pb_curve`: `algorithm × problem → curve = (abscisses, ordinates)`,
plot for each algorithm and problem the associated curve and saves them
as tikz and pdf `figname` file.
"""
function plot_all_curves(algo_pb_curve, figname; FIGS_FOLDER = ".", xmode = "normal")
    plotdata = []

    for (algo, pb_curve) in algo_pb_curve
        plotoptions = StructuredSolvers.get_curve_params(algo.optimizer, 1, 1, 1)

        for (pb, curve) in pb_curve

            ## Attach legend to curve of first probelem, forget otherwise.
            if pb == first(keys(pb_curve))
                push!(
                    plotdata,
                    LegendEntry(StructuredSolvers.get_legendname(algo.optimizer)),
                )
            else
                plotoptions["forget plot"] = nothing
            end
            abscisses_light, ordinates_light =
                simplify_ordinates(curve.abscisses, curve.ordinates)
            push!(
                plotdata,
                PlotInc(
                    PGFPlotsX.Options(plotoptions...),
                    Coordinates(abscisses_light, ordinates_light),
                ),
            )

        end
    end

    fig = @pgf Axis(
        {
            # ymode  = "log",
            xmode  = xmode,
            legend_pos = "outer north east",
            legend_cell_align = "left",
            legend_style = "font=\\footnotesize",
        },
        plotdata...,
    )

    save_fig(fig, figname, FIGS_FOLDER)
    return
end



!(raw"\usepgfplotslibrary{fillbetween}" in PGFPlotsX.CUSTOM_PREAMBLE) && push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")

"""
    plot_aggregateproblems_fillbetween(algo_pb_curve, figname; ...)

Expects a collection `algo_pb_curve`: `algorithm × problem → curve = (abscisses, ordinates)`,
and plots for each algorithm the median of `ordiantes` and a fillbetween of first and third
quartile and saves them as tikz and pdf `figname` file.

Note that median and quartiles can be replaced by other aggregation funcitons, and fillbetween
can be disabled.
"""
function plot_aggregateproblems_fillbetween(
    algo_pb_curve,
    figname;
    FIGS_FOLDER = ".",
    aggregate_pb_perfs = x -> quantile(x, 0.5),
    aggregate_pb_perfs_low = x -> quantile(x, 0.25),
    aggregate_pb_perfs_high = x -> quantile(x, 0.75),
    fillbetween = true
)
    plotdata = []

    algo_id = 0
    for (algo, pb_curve) in algo_pb_curve
        algo_id += 1
        plotoptions = StructuredSolvers.get_curve_params(algo.optimizer, 1, 1, 1)
        algo_legendname = StructuredSolvers.get_legendname(algo.optimizer)

        ##! Assume that all abscisses are the same.
        # * NOTE: adding linear interpolation could deal with this issue
        abscisses = first(values(pb_curve)).abscisses
        for curve in values(pb_curve)
            @assert abscisses == curve.abscisses
        end

        aggregated_ordinates = [
            aggregate_pb_perfs([curve.ordinates[ind] for curve in values(pb_curve)]) for ind in 1:length(abscisses)
        ]

        ## Median curve
        push!(plotdata, PlotInc(
                PGFPlotsX.Options(plotoptions...),
                Coordinates(simplify_ordinates(abscisses, aggregated_ordinates)...),
        ))
        push!(plotdata, LegendEntry(algo_legendname))


        if fillbetween
            ## Plot lower and uppercurves and fillbetween
            aggregated_ordinates_low = [
                aggregate_pb_perfs_low([curve.ordinates[ind] for curve in values(pb_curve)]) for ind in 1:length(abscisses)
            ]
            aggregated_ordinates_high = [
                aggregate_pb_perfs_high([curve.ordinates[ind] for curve in values(pb_curve)]) for ind in 1:length(abscisses)
            ]

            basename = string(algo_id)
            baseoptions = Dict("no_marks"=>nothing, "forget plot"=>nothing, "opacity"=>0)
            push!(plotdata, PlotInc(
                PGFPlotsX.Options(baseoptions..., "name path"=>basename*"-low"),
                Coordinates(simplify_ordinates(abscisses, aggregated_ordinates_low)...)
            ))
            push!(plotdata, PlotInc(
                PGFPlotsX.Options(baseoptions..., "name path"=>basename*"-high"),
                Coordinates(simplify_ordinates(abscisses, aggregated_ordinates_high)...)
            ))
            push!(plotdata, PlotInc(
                PGFPlotsX.Options(
                    "thick" => nothing,
                    "color" => plotoptions["color"],
                    "fill" => plotoptions["color"],
                    "opacity" => 0.4,
                    "forget plot" => nothing
                    ),
                    raw"fill between [of="*basename*"-low and "*basename*"-high]"
            ))
        end

    end

    fig = @pgf Axis(
        {
            # ymode  = "log",
            # xmode  = "log",
            legend_pos = "outer north east",
            legend_cell_align = "left",
            legend_style = "font=\\footnotesize",
        },
        plotdata...,
    )

    save_fig(fig, figname, FIGS_FOLDER)
    return
end


"""
    plot_aggregateproblems_fillbetween(algo_pb_curve, figname; ...)

Expects a collection `algo_pb_curve`: `algorithm × problem → curve = (abscisses, ordinates)`,
plots for each algorithm the median of `ordiantes` and saves them
as tikz and pdf `figname` file.

Note that median can be replaced by other aggregation funcitons such as mean for example.
"""
function plot_aggregateproblems(
    algo_pb_curve,
    figname;
    aggregate_pb_perfs = median,
    FIGS_FOLDER = ".",
)
    return plot_aggregateproblems_fillbetween(
        algo_pb_curve,
        figname;
        FIGS_FOLDER = FIGS_FOLDER,
        aggregate_pb_perfs = aggregate_pb_perfs,
        fillbetween = false
    )
end
