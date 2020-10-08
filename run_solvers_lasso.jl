using StructuredProximalOperators
using CompositeProblems
using StructuredSolvers
using DataStructures
using PGFPlotsX
using Random

function main()


    n, m, sparsity = 100, 60, 0.5
    pb = get_lasso_MLE(n=n, m=m, sparsity=sparsity)
    pbname = "lasso"

    Random.seed!(4567)
    x0 = rand(n)

    optparams = OptimizerParams(
        iterations_limit = 100,
        trace_length = 100,
    )

    # pb = :logit
    # pb = :logit_ionosphere
    pb = :tracenorm

    nit_precisesolve = 500

    if pb == :logit
        n, m, sparsity = 100, 50, 0.5
        pb = get_logit_MLE(n=n, m=m, sparsity=sparsity, λ=0.001)
        pbname = "logit"

        Random.seed!(4567)
        x0 = rand(n)

        optparams = OptimizerParams(
            iterations_limit = 300,
            trace_length = 300,
        )
    elseif pb == :logit_ionosphere
        pb = get_logit_ionosphere(λ=0.01)
        n = problem_dimension(pb)
        pbname = "logit-ionosphere"

        Random.seed!(4567)
        x0 = rand(n)
        maxit = 100

        optparams = OptimizerParams(
            iterations_limit = 60,
            trace_length = 60,
        )
    elseif pb == :tracenorm
        ## TODO...
        n1, n2, m, sparsity = 10, 12, 3, 0.7
        pb = get_tracenorm_MLE(n1=n1, n2=n2, m=m, sparsity=sparsity)
        pbname = "tracenorm"

        Random.seed!(4567)
        x0 = rand(n1, n2)

        nit_precisesolve = 100
    end


    maxit = 500

    @show optparams

    #
    ### Optimal solution
    #
    optimdata = OrderedDict{Optimizer, Any}()


    final_optim_state = StructuredSolvers.precise_solve(pb, x0, iterations_limit=nit_precisesolve)
    x_opt = final_optim_state.additionalinfo.x
    M_opt = final_optim_state.additionalinfo.M
    F_opt = final_optim_state.f_x+final_optim_state.g_x

    display(M_opt)

    # manifold_update = ManifoldTruncatedNewton()
    # optimizer = PartlySmoothOptimizer(manifold_update = manifold_update)

    # initstate = StructuredSolvers.initial_state(optimizer, x0, pb.regularizer)
    # @show fieldnames(typeof(initstate))
    # initstate.update_to_updatestate[manifold_update].νₖ = 1e-20
    # @show initstate.update_to_updatestate[manifold_update].νₖ

    # trace = optimize!(pb, optimizer, x0;
    #     state = initstate,
    #     iterations_limit=200,
    #     trace_length=maxit,
    #     optimstate_extensions=StructuredSolvers.osext
    # )
    # optimdata[optimizer] = trace

    # return

    #
    ### Running algorithms
    #
    # optimdata = OrderedDict{Optimizer, Any}()

    # optimizer = ProximalGradient()
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    # optimizer = ProximalGradient(extrapolation = AcceleratedProxGrad())
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    optimizer = ProximalGradient(extrapolation = MFISTA(AcceleratedProxGrad()))
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace

    # # Alternating
    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldGradient())
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton())
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace

    # ## Adaptive manifold
    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldGradient(), update_selector=ManifoldFollowingSelector())
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton(), update_selector=ManifoldFollowingSelector())
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace


    # Constant manifold
    x0 = project(M_opt, x0)
    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton(), update_selector=ConstantManifoldSelector(M_opt))
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace


    # ## Adaptive manifold

    # ## Constant manifold
    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton(), update_selector=ConstantManifoldSelector(optman))
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace


    #
    ### Build TikzAxis and final plotting object
    #
    fig = TikzDocument()

    push!(fig, TikzPicture(StructuredSolvers.plot_fvals_iteration(optimdata, F_opt=F_opt)))
    push!(fig, TikzPicture(StructuredSolvers.plot_structure_iteration(optimdata, M_opt)))
    push!(fig, TikzPicture(StructuredSolvers.plot_step_iteration(optimdata, F_opt=F_opt)))

    push!(fig, TikzPicture(StructuredSolvers.plot_fvals_time(optimdata, F_opt=F_opt)))
    push!(fig, TikzPicture(StructuredSolvers.plot_structure_time(optimdata, M_opt)))
    push!(fig, TikzPicture(StructuredSolvers.plot_step_time(optimdata, F_opt=F_opt)))

    println("Building output pdf $(pbname) ...")
    PGFPlotsX.pgfsave("figs/$(pbname).pdf", fig)
    PGFPlotsX.pgfsave("figs/$(pbname).tex", fig)

    return fig
end

main()
