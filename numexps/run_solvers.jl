function run_solvers!(optimdata::OrderedDict{Optimizer, Any})

    #
    ### Running algorithms
    #
    optimizer = ProximalGradient()
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace

    optimizer = ProximalGradient(extrapolation = AcceleratedProxGrad())
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace

    # optimizer = ProximalGradient(extrapolation = MFISTA(AcceleratedProxGrad()))
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace
    # optimizer = ProximalGradient(extrapolation = MFISTA(AcceleratedProxGrad()))
    # trace = optimize!(pb, optimizer, trace[end].additionalinfo.x, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace
    # optimizer = ProximalGradient(extrapolation = MFISTA(AcceleratedProxGrad()))
    # trace = optimize!(pb, optimizer, trace[end].additionalinfo.x, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace
    # optimizer = ProximalGradient(extrapolation = MFISTA(AcceleratedProxGrad()))
    # trace = optimize!(pb, optimizer, trace[end].additionalinfo.x, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace



    # Alternating
    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldIdentity())
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldGradient())
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton())
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace


    optimizer = PartlySmoothOptimizer(manifold_update = ManNewtonCG())
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace


    # ## Adaptive manifold
    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldGradient(), update_selector=ManifoldFollowingSelector())
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton(), update_selector=ManifoldFollowingSelector())
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    # # Constant manifold
    # x0 = project(M_opt, x0)
    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton(), update_selector=ConstantManifoldSelector(M_opt))
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace


    # ## Adaptive manifold

    # ## Constant manifold
    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton(), update_selector=ConstantManifoldSelector(optman))
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    return optimdata
end