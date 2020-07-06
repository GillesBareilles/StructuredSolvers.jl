using CompositeProblems
using StructuredSolvers

function get_iterate(state)
    return state.x
end

function main()
    n = 20
    pb = get_lasso(n, 12, 0.6)

    x0 = zeros(n)

    optimizer = ProximalGradient()
    # to, tr = @time optimize!(pb, optimizer, x0)


    # optimizer = ProximalGradient(extrapolation = AcceleratedProxGrad())
    # @time optimize!(pb, optimizer, x0)

    optimstate_extens = [
        (
            key = :iterate,
            getvalue = get_iterate
        ),
    ]

    to, tr = @time optimize!(pb, optimizer, x0, optimstate_extensions = optimstate_extens)

    return tr
end


tr = main()
