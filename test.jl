using CompositeProblems
using StructuredSolvers

function get_iterate(state)
    return state.x
end

function main()
    n = 2
    pb = get_lasso(n, 12, 0.6)
    x0 = zeros(n) .+ 50


    optimstate_extens = [(key = :x, getvalue = get_iterate)]
    optimdata = Dict{Optimizer,OptimizationTrace}()


    optimizer = ProximalGradient()
    to_pg, tr_pg = optimize!(pb, optimizer, x0; optimstate_extensions = optimstate_extens)
    optimdata[optimizer] = tr_pg

    optimizer = ProximalGradient(; extrapolation = AcceleratedProxGrad())
    to_apg, tr_apg = optimize!(pb, optimizer, x0; optimstate_extensions = optimstate_extens)
    optimdata[optimizer] = tr_apg



    ## Compute minimum functional value.
    Fmin_pg = minimum(state.f_x + state.g_x for state in tr_pg)
    Fmin_apg = minimum(state.f_x + state.g_x for state in tr_apg)


    # return StructuredSolvers.plot_fvals_time(optimdata, min(Fmin_pg, Fmin_apg))

    plotit = StructuredSolvers.plot_iterates(pb, optimdata)
    pgfsave("fig.tex", plotit, include_preamble = true)
    return plotit
end


r = main()
