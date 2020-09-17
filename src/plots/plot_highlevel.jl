
"""
    plot_fvals_iteration(optimizer_to_trace; Fmin)

Plot suboptimality as a function of iterations. The baseline functional value should be supplied
for computing suboptimality.
"""
function plot_fvals_iteration(
    optimizer_to_trace::AbstractDict{Optimizer, Any};
    F_opt=nothing,
    state_absciss = s->s.it,
    xlabel="iterations"
)
    if isnothing(F_opt)
        F_opt=+Inf
        for (optimizer, trace) in optimizer_to_trace
            for state in trace
                F_opt = min(F_opt, state.f_x+state.g_x)
            end
        end
    end

    get_abscisses(states) = [state_absciss(state) for state in states]
    function get_ordinates(otpimizer, states)
        return [state.f_x + state.g_x - F_opt for state in states]
    end

    return plot_curves(
        optimizer_to_trace::AbstractDict{Optimizer, Any},
        get_abscisses,
        get_ordinates,
        xlabel = xlabel,
        ylabel = L"$F(x_k)-F^\star$",
        ymode = "log",
        nmarks = 15,
    )
end

function plot_fvals_time(
    optimizer_to_trace::AbstractDict{Optimizer, Any};
    F_opt=nothing
)
    return plot_fvals_iteration(optimizer_to_trace, F_opt=F_opt, state_absciss = s->s.time, xlabel="time (s)")
end


"""
    plot_step_iteration(optimizer_to_trace; Fmin)

Plot suboptimality as a function of iterations. The baseline functional value should be supplied
for computing suboptimality.
"""
function plot_step_iteration(
    optimizer_to_trace::AbstractDict{Optimizer, Any};
    F_opt=nothing,
    state_absciss = s->s.it,
    xlabel = "iterations"
)
    if isnothing(F_opt)
        F_opt=+Inf
        for (optimizer, trace) in optimizer_to_trace
            for state in trace
                F_opt = min(F_opt, state.f_x+state.g_x)
            end
        end
    end

    get_abscisses(states) = [state_absciss(state) for state in states]
    function get_ordinates(otpimizer, states)
        return [state.norm_step for state in states]
    end

    return plot_curves(
        optimizer_to_trace::AbstractDict{Optimizer, Any},
        get_abscisses,
        get_ordinates,
        xlabel = xlabel,
        ylabel = L"$\|x_{k-1}-x_k\|$",
        ymode = "log",
        nmarks = 15,
    )
end
function plot_step_time(
    optimizer_to_trace::AbstractDict{Optimizer, Any};
    F_opt=nothing,
)
    return plot_step_iteration(optimizer_to_trace, F_opt=F_opt, state_absciss = s->s.time, xlabel="time (s)")
end

"""
    plot_structure_iteration(optimizer_to_trace, M_opt)

Plot iterate manifold dimension as a function of iterations.
"""
function plot_structure_iteration(
    optimizer_to_trace::AbstractDict{Optimizer, Any},
    M_opt;
    state_absciss = s->s.it,
    xlabel="iterations"
)
    M_opt_dim = manifold_dimension(M_opt)
    embedding_dim = embedding_dimension(M_opt)

    get_abscisses(states) = [state_absciss(state) for state in states]
    function get_ordinates(optimizer, states)
        # return [100 * (embedding_dim-manifold_dimension(state.additionalinfo.M)) / (embedding_dim - M_opt_dim) for state in states]
        return [manifold_dimension(state.additionalinfo.M) for state in states]
    end

    return plot_curves(
        optimizer_to_trace::AbstractDict{Optimizer, Any},
        get_abscisses,
        get_ordinates,
        xlabel = xlabel,
        ylabel = latexstring("dim(\$M_k\$)"),
        ymode = "normal",
        nmarks = 15,
    )
end
function plot_structure_time(
    optimizer_to_trace::AbstractDict{Optimizer, Any},
    M_opt;
)
    return plot_structure_iteration(optimizer_to_trace, M_opt, state_absciss = s->s.time, xlabel="time (s)")
end






"""
    plot_iterates(pb, optimizer_to_trace)

Plot iterates for 2d problems.
"""
function plot_iterates(pb, optimizer_to_trace::AbstractDict{Optimizer, Any})

    ## TODOs: - automatic / paramters for xmin, xmax, ymin, ymax.
    ## TODOs: - Set colors as in AI paper
    ntraces = length(optimizer_to_trace)
    COLORS = (ntraces <= 7 ? COLORS_7 : COLORS_10)

    xmin, xmax = -1, 4
    ymin, ymax = -1, 2

    plotdata = []

    ## Plot contour
    add_contour!(plotdata, pb, xmin, xmax, ymin, ymax)

    ## Plot algorithms iterates
    algoid = 1
    for (optimizer, trace) in optimizer_to_trace
        coords = [(state.additionalinfo.x[1], state.additionalinfo.x[2]) for state in trace]

        push!(
            plotdata,
            PlotInc(
                PGFPlotsX.Options(get_iteratesplot_params(optimizer, COLORS, algoid)...),
                Coordinates(coords),
            ),
        )
        push!(plotdata, LegendEntry(get_legendname(optimizer)))

        algoid += 1
    end

    # ## Plot optimal point
    # add_point!(ps, xopt)

    ## Plot manifolds
    coords = [(xmin, 0), (xmax, 0)]
    add_manifold!(plotdata, coords)
    coords = [(0, ymin), (0, ymax)]
    add_manifold!(plotdata, coords)

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
            legend_pos = "outer north east",
            legend_cell_align = "left",
            legend_style = "font=\\footnotesize",
            # title = "Problem $(pb.name) -- Iterates postion",
        },
        plotdata...,
    )
end
