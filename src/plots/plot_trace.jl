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


get_legendname(optimizer) = Base.summary(optimizer)

function get_curve_params(optimizer, COLORS, algoid, markrepeat)
    return Dict{Any,Any}(
        "mark" => MARKERS[mod(algoid, 7) + 1],
        "color" => COLORS[algoid],
        "mark repeat" => markrepeat,
        # "mark phase" => 7,
        # "mark options" => "draw=black",
    )
end

function plot_curves(
    optimizer_to_trace::AbstractDict{Optimizer, Any},
    get_abscisses,
    get_ordinates;
    xlabel = "time (s)",
    ylabel = "",
    ymode = "log",
    nmarks = 15,
    includelegend = true,
)
    plotdata = []
    ntraces = length(optimizer_to_trace)
    COLORS = (ntraces <= 7 ? COLORS_7 : COLORS_10)

    maxloggedvalues = maximum(length(trace) for trace in values(optimizer_to_trace))
    markrepeat = floor(maxloggedvalues / nmarks)

    algoid = 1
    for (optimizer, trace) in optimizer_to_trace
        push!(
            plotdata,
            PlotInc(
                PGFPlotsX.Options(get_curve_params(
                    optimizer,
                    COLORS,
                    algoid,
                    markrepeat,
                )...),
                Coordinates(get_abscisses(trace), get_ordinates(optimizer, trace)),
            ),
        )
        includelegend && push!(plotdata, LegendEntry(get_legendname(optimizer)))
        algoid += 1
    end
    return @pgf Axis(
        {
            ymode = ymode,
            # height = "10cm",
            # width = "10cm",
            xlabel = xlabel,
            ylabel = ylabel,
            legend_pos = "outer north east",
            legend_style = "font=\\footnotesize",
            legend_cell_align = "left",
            xmin = 0,
        },
        plotdata...,
    )
end





#
## Helper functions.
#
function get_iteratesplot_params(optimizer, COLORS, algoid)
    return Dict{Any,Any}(
        "mark" => MARKERS[mod(algoid, 7) + 1],
        "color" => COLORS[algoid],
        # "mark phase" => 7,
        # "mark options" => "draw=black",
    )
end


function add_contour!(plotdata, pb, xmin, xmax, ymin, ymax)
    x = xmin:((xmax - xmin) / 100):xmax
    y = ymin:((ymax - ymin) / 100):ymax
    F = (x, y) -> f(pb, [x, y])

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
