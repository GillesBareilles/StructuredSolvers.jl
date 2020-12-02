using StructuredProximalOperators
using CompositeProblems
using StructuredSolvers
using DataStructures
using PGFPlotsX
using Random
using LinearAlgebra
using Distributions

include("output_table_fig.jl")


function main()
    n, m, sparsity = 100, 130, 0.5
    # n, m, sparsity = 10, 11, 0.5
    pb = get_lasso_MLE(n=n, m=m, sparsity=sparsity)
    pbname = "lasso"

    x0 = rand(n) .* 0

    optparams = OptimizerParams(
        iterations_limit = 100,
        trace_length = 100,
    )

    #
    ### Optimal solution
    #
    # final_optim_state = StructuredSolvers.precise_solve(pb, x0, iterations_limit=3)
    # x_opt = final_optim_state.additionalinfo.x
    # M_opt = final_optim_state.additionalinfo.M
    # F_opt = final_optim_state.f_x+final_optim_state.g_x
    # display(M_opt)


    #
    ### Running algorithms
    #
    optimdata = OrderedDict{Optimizer, Any}()

    optimizer = ProximalGradient()
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace

    optimizer = ProximalGradient(extrapolation = AcceleratedProxGrad())
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace

    optimizer = ProximalGradient(extrapolation = MFISTA(AcceleratedProxGrad()))
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace

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


    M_opt = last(trace).additionalinfo.M
    F_opt = last(trace).f_x+last(trace).g_x


    # Constant manifold
    x0 = project(M_opt, x0)
    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton(), update_selector=ConstantManifoldSelector(M_opt))
    trace = optimize!(pb, optimizer, x0, manifold=M_opt, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace




    #
    ### Build table
    #
    build_table(optimdata, pbname, M_opt=M_opt, F_opt=F_opt)

    #
    ### Build TikzAxis and final plotting object
    #
    fig = plot_numexps_data(optimdata, pbname, M_opt)

    return fig
end

main()
