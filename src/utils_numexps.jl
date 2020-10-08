get_iterate(state) = state.x
get_manifold(state) = state.M

osext = [
    (key = :x, getvalue = get_iterate),
    (key = :M, getvalue = get_manifold)
]

function precise_solve(pb, x0; iterations_limit=400)
    println("-- Precise solve")

    optparams = OptimizerParams(
        iterations_limit = iterations_limit,
        show_trace=false,
    )

    optimizer = ProximalGradient(extrapolation = MFISTA(AcceleratedProxGrad()))
    trace = optimize!(pb, optimizer, x0, optimstate_extensions=osext, optparams=optparams)
    @show trace[end].norm_step


    optimizer = ProximalGradient(extrapolation = MFISTA(AcceleratedProxGrad()))
    trace = optimize!(pb, optimizer, trace[end].additionalinfo.x, optimstate_extensions=osext, optparams=optparams)
    @show trace[end].norm_step

    return trace[end]
end
