

COLORS_7 = [
    Colors.RGB(68 / 255, 119 / 255, 170 / 255),
    Colors.RGB(102 / 255, 204 / 255, 238 / 255),
    Colors.RGB(34 / 255, 136 / 255, 51 / 255),
    Colors.RGB(204 / 255, 187 / 255, 68 / 255),
    Colors.RGB(238 / 255, 102 / 255, 119 / 255),
    Colors.RGB(170 / 255, 51 / 255, 119 / 255),
    Colors.RGB(187 / 255, 187 / 255, 187 / 255),
]

COLORS_10 = [
    colorant"#332288",
    colorant"#88CCEE",
    colorant"#44AA99",
    colorant"#117733",
    colorant"#999933",
    colorant"#DDCC77",
    colorant"#CC6677",
    colorant"#882255",
    colorant"#AA4499",
    colorant"#DDDDDD",
]

MARKERS = ["x", "+", "star", "oplus", "triangle", "diamond", "pentagon"]



function plot_curves(
    optimizer_to_trace::Dict{Optimizer,OptimizationTrace},
    get_abscisses,
    get_ordinates;
    xlabel = "time (s)",
    ylabel = "",
    ymode = "log",
    nmarks = 15,
)
    plotdata = []
    ntraces = length(optimizer_to_trace)
    COLORS = (ntraces <= 7 ? COLORS_7 : COLORS_10)

    maxlogedvalues = maximum(length(trace) for trace in values(optimizer_to_trace))
    markrepeat = floor(maxlogedvalues / nmarks)

    algoid = 1
    for (optimizer, trace) in optimizer_to_trace
        push!(
            plotdata,
            PlotInc(
                PGFPlotsX.Options(
                    "mark" => MARKERS[mod(algoid, 7) + 1],
                    "color" => COLORS[algoid],
                    "mark repeat" => markrepeat,
                    # "mark phase" => 7,
                    # "mark options" => "draw=black",
                ),
                Coordinates(get_abscisses(trace), get_ordinates(trace)),
            ),
        )
        push!(plotdata, LegendEntry(replace(string(optimizer), "-" => " ")))
        algoid += 1
    end
    return @pgf Axis(
        {
            ymode = ymode,
            height = "10cm",
            width = "10cm",
            xlabel = xlabel,
            ylabel = ylabel,
            legend_pos = "north east",
            legend_cell_align = "left",
            xmin = 0,
        },
        plotdata...,
    )
end





"""
    plot_fvals_time(optimizer_to_trace; Fmin)

Plot suboptimality as a function of time. The baseline functional value should be supplied
for computing suboptimality.
"""
function plot_fvals_time(optimizer_to_trace::Dict{Optimizer,OptimizationTrace}, Fmin)
    get_abscisses(states) = [state.it for state in states]
    function get_ordinates(states)
        return [state.f_x + state.g_x - Fmin for state in states]
    end

    return plot_curves(
        optimizer_to_trace::Dict{Optimizer,OptimizationTrace},
        get_abscisses,
        get_ordinates,
        xlabel = "time (s)",
        ylabel = "",
        ymode = "log",
        nmarks = 15,
    )
end





"""
    plot_iterates(pb, optimizer_to_trace)

Plot iterates for 2d problems.
"""
function plot_iterates(pb, optimizer_to_trace::Dict{Optimizer,OptimizationTrace})

    ## TODOs: - automatic / paramters for xmin, xmax, ymin, ymax.
    ## TODOs: - Set colors as in AI paper
    xmin, xmax = -1, 4
    ymin, ymax = -1, 2

    plotdata = []

    ## Plot contour
    add_contour!(plotdata, pb, xmin, xmax, ymin, ymax)

    ## Plot algorithms iterates
    for (optimizer, trace) in optimizer_to_trace
        coords = [(state.additionalinfo.x[1], state.additionalinfo.x[2]) for state in trace]

        push!(
            plotdata,
            PlotInc(
                PGFPlotsX.Options(Dict(
                    # "mark options"=>nothing,
                    "smooth" => nothing,
                    "mark size" => "{1pt}",# color={rgb,1:red,0.0;green,0.3843;blue,0.9922},
                    "mark" => "*",#, very thick]
                )...),
                Coordinates(coords),
            ),
        )
        push!(plotdata, LegendEntry(replace(string(optimizer), "-" => " ")))
    end



    # ## Plot optimal point
    # add_point!(ps, xopt)


    ## Plot manifolds
    coords = [(xmin, 0), (xmax, 0)]
    add_manifold!(plotdata, coords)
    coords = [(0, ymin), (0, ymax)]
    add_manifold!(plotdata, coords)
    @show plotdata[1]
    @show typeof(plotdata[1])

    return @pgf Axis(
        {
            contour_prepared,
            # view = "{0}{90}",
            # height = "12cm",
            # width = "12cm",
            xmin = xmin,
            xmax = xmax,
            ymin = ymin,
            ymax = ymax,
            legend_pos = "north east",
            legend_cell_align = "left",
            legend_style = "font=\\footnotesize",
            # title = "Problem $(pb.name) -- Iterates postion",
        },
        plotdata...,
    )

    return fig = pgf_build_iteratesfig(ps, xmin, xmax, ymin, ymax)
end


#
## Helper functions.
#


function add_contour!(plotdata, pb, xmin, xmax, ymin, ymax)
    x = xmin:((xmax - xmin) / 100):xmax
    y = ymin:((ymax - ymin) / 100):ymax
    F = (x, y) -> f(pb, [x, y]) + g(pb, [x, y])

    @assert problem_dimension(pb) == 2 "Problem should be two dimensional."

    push!(
        plotdata,
        PlotInc(
            PGFPlotsX.Options(
                "forget plot" => nothing,
                "no marks" => nothing,
                "ultra thin" => nothing,
            ),
            Table(contours(x, y, F.(x, y'), 10)),
        ),
    )
    return
end

function add_point!(plotdata, xopt)
    coords = [(xopt[1], xopt[2])]

    push!(
        plotdata,
        PlotInc(
            PGFPlotsX.Options(
                "forget plot" => nothing,
                "only marks" => nothing,
                "mark" => "star",
                "thick" => nothing,
                "color" => "black",
            ),
            Coordinates(coords),
        ),
    )

    return
end

function add_manifold!(ps, coords)
    push!(
        ps,
        PlotInc(
            PGFPlotsX.Options(
                "forget plot" => nothing,
                "no marks" => nothing,
                "smooth" => nothing,
                "thick" => nothing,
                "solid" => nothing,
                "black!50!white" => nothing,
                # "mark size" => "1pt"
            ),
            Coordinates(coords),
        ),
    )
    return
end
