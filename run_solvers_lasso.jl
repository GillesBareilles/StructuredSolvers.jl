using StructuredProximalOperators
using CompositeProblems
using StructuredSolvers
using DataStructures

function main()

    n = 200
    pb = get_random_qualifiedleastsquares(n, 120, regularizer_l1(1), 0.6)

    x0 = zeros(n) .+1

    optimdata = OrderedDict{Optimizer, Any}()

    optimizer = ProximalGradient()
    trace = optimize!(pb, optimizer, x0, iterations_limit=75)
    optimdata[optimizer] = trace

    optimizer = ProximalGradient(extrapolation = AcceleratedProxGrad())
    trace = optimize!(pb, optimizer, x0, iterations_limit=75)
    optimdata[optimizer] = trace

    optimizer = PartlySmoothOptimizer()
    trace = optimize!(pb, optimizer, x0, iterations_limit=75)
    optimdata[optimizer] = trace

    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldGradient())
    trace = optimize!(pb, optimizer, x0, iterations_limit=75)
    optimdata[optimizer] = trace

    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton())
    trace = optimize!(pb, optimizer, x0, iterations_limit=75)
    optimdata[optimizer] = trace


    ## Adaptive manifold
    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldGradient(), update_selector=ManifoldFollowingSelector())
    trace = optimize!(pb, optimizer, x0, iterations_limit=75)
    optimdata[optimizer] = trace

    ## Constant manifold
    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldGradient(), update_selector=ConstantManifoldSelector(l1Manifold(ones(n))))
    trace = optimize!(pb, optimizer, x0, iterations_limit=75)
    optimdata[optimizer] = trace

    ## Adaptive manifold
    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton(), update_selector=ManifoldFollowingSelector())
    trace = optimize!(pb, optimizer, x0, iterations_limit=75)
    optimdata[optimizer] = trace

    ## Constant manifold
    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton(), update_selector=ConstantManifoldSelector(l1Manifold(ones(n))))
    trace = optimize!(pb, optimizer, x0, iterations_limit=75)
    optimdata[optimizer] = trace

    plot_subopt = StructuredSolvers.plot_fvals_iteration(optimdata)
    # save("subopt.tex", plot_subopt, include_preamble = true)
    # save("subopt.pdf", plot_subopt)

    # return
end

main()
