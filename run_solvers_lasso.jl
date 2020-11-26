using StructuredProximalOperators
using CompositeProblems
using StructuredSolvers
using DataStructures
using PGFPlotsX
using Random
using LinearAlgebra
using Distributions

function main()


    n, m, sparsity = 100, 130, 0.5
    # n, m, sparsity = 10, 10, 0.5
    pb = get_lasso_MLE(n=n, m=m, sparsity=sparsity)
    pbname = "lasso"

    Random.seed!(4567)
    x0 = rand(n) .* 0

    optparams = OptimizerParams(
        iterations_limit = 35,
        trace_length = 35,
    )

    @show pb
    @show optparams

    #
    ### Optimal solution
    #
    final_optim_state = StructuredSolvers.precise_solve(pb, x0, iterations_limit=3)
    x_opt = final_optim_state.additionalinfo.x
    M_opt = final_optim_state.additionalinfo.M
    F_opt = final_optim_state.f_x+final_optim_state.g_x
    display(M_opt)


    #
    ### Running algorithms
    #
    optimdata = OrderedDict{Optimizer, Any}()

    optimizer = ProximalGradient()
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace

    # optimizer = ProximalGradient(extrapolation = AcceleratedProxGrad())
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

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

    # @show trace[end].additionalinfo.x

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


    #
    ### Build TikzAxis and final plotting object
    #
    fig = TikzDocument()

    push!(fig, TikzPicture(StructuredSolvers.plot_fvals_iteration(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_structure_iteration(optimdata, M_opt)))
    push!(fig, TikzPicture(StructuredSolvers.plot_step_iteration(optimdata)))

    push!(fig, TikzPicture(StructuredSolvers.plot_fvals_time(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_structure_time(optimdata, M_opt)))
    push!(fig, TikzPicture(StructuredSolvers.plot_step_time(optimdata)))

    println("Building output pdf $(pbname) ...")
    PGFPlotsX.pgfsave("figs/$(pbname).pdf", fig)
    PGFPlotsX.pgfsave("figs/$(pbname).tex", fig)

    return fig
end

main()
