using CompositeProblems
using StructuredSolvers

function main()
    n = 20
    pb = get_lasso(n, 12, 0.6)

    x0 = zeros(n)

    optimizer = ProximalGradient()
    @time optimize!(pb, optimizer, x0)

    optimizer = ProximalGradient(extrapolation = AcceleratedProxGrad())
    @time optimize!(pb, optimizer, x0)
    return
end


main()
