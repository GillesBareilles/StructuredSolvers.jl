
# function add_trace!(plotdata, optimization_traces::Vector{OptimizationTrace})

#     plot options, legendname = 1, 2


#     push!(plotdata, PlotInc(
#         plotoptions,
#         Coordinates(global_plotdata[kdisplayeddata].time, global_plotdata[kdisplayeddata].medianvalue)
#     ))
#     push!(plotdata, LegendEntry(legendname))
#     return
# end

function plot(optimizer_to_trace::Dict{Optimizer, OptimizationTrace}, get_absciss, get_ordinate)
    plotdata = []

    for (optimizer, trace) in optimizer_to_trace
        push!(plotdata, PlotInc(
            plotoptions,
            Coordinates(
                [get_absciss(state) for state in trace],
                [get_ordinate(state) for state in trace],
        )))
        push!(plotdata, LegendEntry(legendname))
    end
    return @pgf Axis(
        {
            ymode = vdisplayeddata.ymode,
            height = "10cm",
            width = "10cm",
            xlabel = "time (s)",
            ylabel = vdisplayeddata.ylabel,
            legend_pos = "north east",
            legend_cell_align = "left",
            xmin = 0,
        },
        plotdata...,
    )
end







## Iterate over kinds of plots wrt algos displayed
for (plot_exten, fnsolve_whitelist) in plotname_to_fnsolve, (kdisplayeddata, vdisplayeddata) in plot_to_displayeddata
    plotdata = []

    printstyled("\n-- $plot_exten ; ", string(kdisplayeddata), "\n", color=:red)

    for (global_key, global_plotdata) in global_data
        cur_fnsolve_symbol, cur_nworkers, cur_sampling_distr, cur_stepsize = global_key

        !(cur_fnsolve_symbol in fnsolve_whitelist) && continue
        printstyled("Adding $problem_name, $cur_fnsolve_symbol, $cur_nworkers, $cur_sampling_distr, $cur_stepsize...\n", color=:blue)

        plotoptions, legendname = get_plotstyle(global_key)

        # Median curve
        push!(plotdata, PlotInc(
            plotoptions,
            Coordinates(global_plotdata[kdisplayeddata].time, global_plotdata[kdisplayeddata].medianvalue)
        ))
        push!(plotdata, LegendEntry(legendname))

        ## Plot lower and uppercurves and fillbetween
        basepathname = string(cur_fnsolve_symbol, "-", cur_nworkers, "-", cur_sampling_distr, "-", cur_stepsize)
        baseopts = Dict("no_marks"=>nothing, "forget plot"=>nothing, "opacity"=>0)
        push!(plotdata, PlotInc(
            PGFPlotsX.Options(baseopts..., "name path"=>basepathname*"-low"),
            Coordinates(global_plotdata[kdisplayeddata].time, global_plotdata[kdisplayeddata].firstquartile)
        ))
        push!(plotdata, PlotInc(
            PGFPlotsX.Options(baseopts..., "name path"=>basepathname*"-high"),
            Coordinates(global_plotdata[kdisplayeddata].time, global_plotdata[kdisplayeddata].thirdquartile)
        ))
        push!(plotdata, PlotInc(
            PGFPlotsX.Options(
                "thick" => nothing,
                "color" => plotoptions["color"],
                "fill" => plotoptions["color"],
                "opacity" => 0.4,
                "forget plot" => nothing
                ),
                raw"fill between [of="*basepathname*"-low and "*basepathname*"-high]"
        ))

    end

    fig_time_subopt = @pgf Axis(
        {
            ymode = vdisplayeddata.ymode,
            height = "10cm",
            width = "10cm",
            xlabel = "time (s)",
            ylabel = vdisplayeddata.ylabel,
            legend_pos = "north east",
            legend_cell_align = "left",
            xmin = 0,
            # xmax = 100,
            # ymin = 1e-10,
            # ymax = 1,
        },
        plotdata...,
    )

    ## Plotting section
    FIGS_FOLDER = logfolder
    FIGS_FOLDER = "./temp"

    println()
    filename = setupname*"_"*plot_exten*"_"*string(kdisplayeddata)
    println("Saving: ", filename*".tex")
    pgfsave(joinpath(FIGS_FOLDER, filename*".tex"), fig_time_subopt, include_preamble=false)
    println("Saving: ", filename*".pdf")
    pgfsave(joinpath(FIGS_FOLDER, filename*".pdf"), fig_time_subopt)
