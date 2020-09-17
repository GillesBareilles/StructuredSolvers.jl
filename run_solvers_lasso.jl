using StructuredProximalOperators
using CompositeProblems
using StructuredSolvers
using DataStructures
using PGFPlotsX
using Random

function main()


    n, m, sparsity = 100, 20, 0.5
    pb = get_lasso_MLE(n=n, m=m, sparsity=sparsity)
    pbname = "lasso"

    Random.seed!(4567)
    x0 = rand(n)


    pb = :logit
    pb = :logit_ionosphere
    # pb = :tracenorm

    if pb == :logit
        n, m, sparsity = 100, 50, 0.5
        pb = get_logit_MLE(n=n, m=m, sparsity=sparsity, λ=0.001)
        pbname = "logit"

        Random.seed!(4567)
        x0 = rand(n)
    elseif pb == :logit_ionosphere
        pb = get_logit_ionosphere()
        n = problem_dimension(pb)
        pbname = "logit-ionosphere"

        Random.seed!(4567)
        x0 = rand(n)
    elseif pb == :tracenorm
        ## TODO...
        n1, n2, m, sparsity = 15, 17, 50, 0.5
        pb = get_tracenorm_MLE(n1=n1, n2=n2, m=m, sparsity=sparsity)
        pbname = "tracenorm"

        Random.seed!(4567)
        x0 = rand(n)
    end


    maxit = 500

    #
    ### Optimal solution
    #
    optimdata = OrderedDict{Optimizer, Any}()


    final_optim_state = StructuredSolvers.precise_solve(pb, x0, iterations_limit=50000)
    x_opt = final_optim_state.additionalinfo.x
    M_opt = final_optim_state.additionalinfo.M
    F_opt = final_optim_state.f_x+final_optim_state.g_x

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

    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton())
    trace = optimize!(pb, optimizer, x0, iterations_limit=maxit, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace


    ## Adaptive manifold
    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldGradient(), update_selector=ManifoldFollowingSelector())
    trace = optimize!(pb, optimizer, x0, iterations_limit=maxit, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace

    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton(), update_selector=ManifoldFollowingSelector())
    trace = optimize!(pb, optimizer, x0, iterations_limit=maxit, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace


    # Constant manifold
    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldGradient(), update_selector=ConstantManifoldSelector(M_opt))
    # trace = optimize!(pb, optimizer, x0, iterations_limit=maxit, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    # ## Adaptive manifold

    # ## Constant manifold
    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton(), update_selector=ConstantManifoldSelector(optman))
    # trace = optimize!(pb, optimizer, x0, iterations_limit=maxit, optimstate_extensions=StructuredSolvers.osext)
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
