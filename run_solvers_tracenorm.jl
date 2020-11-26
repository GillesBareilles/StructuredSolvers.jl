using StructuredProximalOperators
using CompositeProblems
using StructuredSolvers
using DataStructures
using PGFPlotsX
using Random
using LinearAlgebra
using Distributions

function main()
    ########################
    # Tough nuclear problem
    n1, n2, m, sparsity = 3, 3, 1, 0.8
    seed = 1234
    δ=0.01

    A = Vector{Matrix{Float64}}(undef, m)
    for i in 1:m
        Random.seed!(seed+i)
        A[i] = rand(Normal(), n1, n2)
    end

    @show A
    display(A[1])

    ## Generating structured signal
    nsingvals = min(n1, n2)
    optstructure = FixedRankMatrices(n1, n2, 1)

    Random.seed!(seed-1)
    x0 = project(optstructure, rand(Normal(), n1, n2).*100)
    @show x0.U
    @show x0.S
    @show x0.Vt

    x0_emb = embed(optstructure, x0)
    @show rank(x0_emb)

    ## Noised measurements
    Random.seed!(seed)
    e = rand(Normal(0, δ^2), m)

    y = dot(A[1], x0) .+ e

    pb = TracenormPb(A, y, n1, n2, 1, regularizer_lnuclear(δ), x0, optstructure)
    ########################

    pbname = "tracenorm-3x3"

    nit_precisesolve = 10


    optparams = OptimizerParams(
        iterations_limit = 200,
        trace_length = 200,
    )

    @show pb
    @show optparams

    #
    ### Optimal solution
    #
    optimdata = OrderedDict{Optimizer, Any}()

    x0 = [
        0.206225  -0.363245  -0.0280843
        -0.799533   1.4083     0.108883
         0.649714  -1.1444    -0.0884798
    ]

    x0 = zeros(3, 3)


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
    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldIdentity())
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace

    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldGradient())
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton())
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
