using StructuredProximalOperators
using CompositeProblems
using StructuredSolvers
using DataStructures
using PGFPlotsX

function main()

    n, m, sparsity = 10, 5, 0.5
    pb = get_lasso_MLE(n=n, m=m, sparsity=sparsity)

    pbname = "lasso"

    x0 = zeros(n) .+1
    # x0 = pb.x0 .+ 1e-5

    maxit = 200

    #
    ### Optimal solution
    #
    final_optim_state = StructuredSolvers.precise_solve(pb, x0)
    x_opt = final_optim_state.additionalinfo.x
    M_opt = final_optim_state.additionalinfo.M
    F_opt = final_optim_state.f_x+final_optim_state.g_x

    #
    ### Running algorithms
    #
    optimdata = OrderedDict{Optimizer, Any}()

    optimizer = ProximalGradient()
    trace = optimize!(pb, optimizer, x0, iterations_limit=maxit, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace

    optimizer = ProximalGradient(extrapolation = AcceleratedProxGrad())
    trace = optimize!(pb, optimizer, x0, iterations_limit=maxit, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace

    optimizer = ProximalGradient(extrapolation = MFISTA(AcceleratedProxGrad()))
    trace = optimize!(pb, optimizer, x0, iterations_limit=maxit, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace

    # optimizer = ProximalGradient(extrapolation = RestartedAPG())
    # trace = optimize!(pb, optimizer, x0, iterations_limit=maxit, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace


    # Alternating
    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldGradient())
    trace = optimize!(pb, optimizer, x0, iterations_limit=maxit, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace

    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton())
    # trace = optimize!(pb, optimizer, x0, iterations_limit=maxit, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace


    # ## Adaptive manifold
    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldGradient(), update_selector=ManifoldFollowingSelector())
    # trace = optimize!(pb, optimizer, x0, iterations_limit=maxit, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    ## Constant manifold
    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldGradient(), update_selector=ConstantManifoldSelector(optman))
    # trace = optimize!(pb, optimizer, x0, iterations_limit=maxit, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    # ## Adaptive manifold
    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton(), update_selector=ManifoldFollowingSelector())
    # trace = optimize!(pb, optimizer, x0, iterations_limit=maxit, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    # ## Constant manifold
    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton(), update_selector=ConstantManifoldSelector(optman))
    # trace = optimize!(pb, optimizer, x0, iterations_limit=maxit, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace


    #
    ### Build TikzAxis and final plotting object
    #
    plot_subopt = StructuredSolvers.plot_fvals_iteration(optimdata, F_opt=F_opt)
    plot_structure = StructuredSolvers.plot_structure_iteration(optimdata, M_opt)
    plot_step = StructuredSolvers.plot_step_iteration(optimdata, F_opt=F_opt)

    fig = TikzDocument(TikzPicture(plot_subopt), TikzPicture(plot_structure), TikzPicture(plot_step))
    PGFPlotsX.pgfsave("figs/$(pbname).pdf", fig)
    PGFPlotsX.pgfsave("figs/$(pbname).tex", fig)

    return fig
end

main()
