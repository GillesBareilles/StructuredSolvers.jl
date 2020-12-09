using StructuredProximalOperators
using CompositeProblems
using StructuredSolvers
using DataStructures
using PGFPlotsX
using Random
using LinearAlgebra
using Distributions

function main()
    pb = :logit
    pb = :logit_ionosphere

    nit_precisesolve = 500
    nit_precisesolve = 3

    if pb == :logit
        n, m, sparsity = 100, 50, 0.5
        pb = get_logit_MLE(n=n, m=m, sparsity=sparsity, λ=0.001)
        pbname = "logit"

        Random.seed!(4567)
        x0 = zeros(n)

        optparams = OptimizerParams(
            iterations_limit = 150,
            trace_length = 150,
        )
    elseif pb == :logit_ionosphere
        pb = get_logit_ionosphere(λ=0.01)
        n = problem_dimension(pb)
        pbname = "logit-ionosphere"

        Random.seed!(4567)
        x0 = zeros(n)
        maxit = 100

        optparams = OptimizerParams(
            iterations_limit = 60,
            trace_length = 60,
        )
    end

    @show pb
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


    optparams = OptimizerParams(
        iterations_limit = 200,
        trace_length = 200,
    )


    #
    ### Running algorithms
    #
    # optimdata = OrderedDict{Optimizer, Any}()

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

    optimizer = PartlySmoothOptimizer(manifold_update = ManTruncatedNewton())
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace


    optimizer = PartlySmoothOptimizer(manifold_update = ManNewtonCG())
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace

    # return
    # ## Adaptive manifold
    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldGradient(), update_selector=ManifoldFollowingSelector())
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    # optimizer = PartlySmoothOptimizer(manifold_update = ManTruncatedNewton(), update_selector=ManifoldFollowingSelector())
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    # # Constant manifold
    # x0 = project(M_opt, x0)
    # optimizer = PartlySmoothOptimizer(manifold_update = ManTruncatedNewton(), update_selector=ConstantManifoldSelector(M_opt))
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace


    # ## Adaptive manifold

    # ## Constant manifold
    # optimizer = PartlySmoothOptimizer(manifold_update = ManTruncatedNewton(), update_selector=ConstantManifoldSelector(optman))
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace


    #
    ### Build TikzAxis and final plotting object
    #
    fig = TikzDocument()

    # F_opt = 2.5397380626720314e+00

    push!(fig, TikzPicture(StructuredSolvers.plot_fvals_iteration(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_tangentres_iteration(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_normalres_iteration(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_structure_iteration(optimdata, M_opt)))
    push!(fig, TikzPicture(StructuredSolvers.plot_step_iteration(optimdata)))

    push!(fig, TikzPicture(StructuredSolvers.plot_fvals_time(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_tangentres_time(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_normalres_time(optimdata)))
    push!(fig, TikzPicture(StructuredSolvers.plot_structure_time(optimdata, M_opt)))
    push!(fig, TikzPicture(StructuredSolvers.plot_step_time(optimdata)))

    println("Building output pdf $(pbname) ...")
    PGFPlotsX.pgfsave("figs/$(pbname).pdf", fig)
    PGFPlotsX.pgfsave("figs/$(pbname).tex", fig)

    return fig
end

main()
