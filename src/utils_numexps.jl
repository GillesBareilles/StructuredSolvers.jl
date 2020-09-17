get_iterate(state) = state.x
get_manifold(state) = state.M

osext = [
    (key = :x, getvalue = get_iterate),
    (key = :M, getvalue = get_manifold)
]

function precise_solve(pb, x0; iterations_limit=400)
    optimizer = ProximalGradient(extrapolation = MFISTA(AcceleratedProxGrad()))
    trace = optimize!(pb, optimizer, x0, iterations_limit=iterations_limit, optimstate_extensions=osext, show_trace=false)

    optimizer = ProximalGradient(extrapolation = MFISTA(AcceleratedProxGrad()))
    trace = optimize!(pb, optimizer, trace[end].additionalinfo.x, iterations_limit=iterations_limit, optimstate_extensions=osext, show_trace=false)

    return trace[end]
end
