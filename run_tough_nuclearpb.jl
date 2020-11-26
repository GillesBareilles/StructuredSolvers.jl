using StructuredProximalOperators
using CompositeProblems
using StructuredSolvers
using DataStructures
using PGFPlotsX
using Random
using LinearAlgebra
using Distributions

function tough_small_nuclear_pb()
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
    x0 = project(optstructure, rand(Normal(), n1, n2))
    @show x0.U
    @show x0.S
    @show x0.Vt

    x0_emb = embed(optstructure, x0)
    @show rank(x0_emb)

    ## Noised measurements
    Random.seed!(seed)
    e = rand(Normal(0, δ^2), m)

    y = dot(A[1], x0) .+ e

    return TracenormPb(A, y, n1, n2, 1, regularizer_lnuclear(δ), x0, optstructure)
end

function examine_f_direction_TruncatedNewton(pb, M, x)
    x = project(M, x)

    ∇f_x = ∇f(pb, x)

    grad_fgₖ = egrad_to_rgrad(M, x, ∇f_x) + ∇M_g(pb, M, x)
    function hessfg_x_h(ξ)
        return ehess_to_rhess(M, x, ∇f_x, ∇²f_h(pb, x, ξ), ξ) + ∇²M_g_ξ(pb, M, x, ξ)
    end

    norm_rgrad = norm(M, x, grad_fgₖ)
    @show norm_rgrad

    check_tangent_vector(M, x, grad_fgₖ)

    ## 2. Get Truncated Newton direction
    ϵ_residual = min(0.5, sqrt(norm_rgrad)) * norm_rgrad    # Forcing sequence as sugested in NW, p. 168
    ϵ_residual = 1e-13
    νₖ = 1e-10
    maxiter = 40
    dᴺ, CG_niter, d_type = StructuredSolvers.solve_tCG(M, x, grad_fgₖ, hessfg_x_h, ϵ_residual = ϵ_residual, ν = νₖ, printlev=0, maxiter=maxiter)

    @show inner(M, x, grad_fgₖ, dᴺ)
    @show inner(M, x, grad_fgₖ, dᴺ) / (norm(M, x, grad_fgₖ) * norm(M, x, dᴺ))


    ## Curves section
    η = dᴺ

    f_x = f(pb, x)
    ηgradfx = inner(M, x, η, grad_fgₖ)
    ηHessf_xη = inner(M, x, η, hessfg_x_h(η))

    @show ηgradfx, ηHessf_xη

    frgrad(t) = abs(f(pb, retract(M, x, t * η)) - (f_x + t * ηgradfx))
    function frhess(t)
        return abs(
            f(pb, retract(M, x, t * η)) - (f_x + t * ηgradfx + 0.5 * t^2 * ηHessf_xη),
        )
    end

    curves_comp = compare_curves(frgrad, frhess)


    tmin, tmax = 1e-8, 1e8
    npoints = 50
    ts = 10 .^ range(log(10, tmin), stop = log(10, tmax), length = npoints)
    curves = Dict(
        :f => []
    )

    for (i, t) in enumerate(ts)
        val = f(pb, retract(M, x, t*η))
        push!(curves[:f], (t, val))

        # push!(curves[:slope2], (t, t^2))
        # push!(curves[:slope3], (t, t^3))
    end
    curves_comp = curves

    @show (curves)


    return display_curvescomparison(curves_comp)
end



function main()



    nit_precisesolve = 500
    nit_precisesolve = 3

    #
    ### Tough nuclear problem
    #
    pb = tough_small_nuclear_pb()

    pbname = "tracenorm"

    # Random.seed!(4567)
    # x0 = rand(n1, n2)

    nit_precisesolve = 10

    optimdata = OrderedDict{Optimizer, Any}()


    #
    ### Optimal solution
    #
    # We start close to the minimizer
    x0 = [
        0.206225  -0.363245  -0.0280843
        -0.799533   1.4083     0.108883
         0.649714  -1.1444    -0.0884798
    ]

    # x_appsol  = [0.2062254145171265 -0.36324489509328695 -0.028084370680129245; -0.7995328241862998 1.4082949840358918 0.10888268188455438; 0.6497143153479746 -1.144405062158729 -0.0884799660174937]
    x_appsol2 = [0.2062254133154103 -0.3632448975037345 -0.02808436995755387; -0.799532820766673 1.4082949955642523 0.10888267925192681; 0.6497143041477935 -1.144405056693562 -0.0884799627313333]

    final_optim_state = StructuredSolvers.precise_solve(pb, x_appsol2, iterations_limit=nit_precisesolve)
    x_opt = final_optim_state.additionalinfo.x
    M_opt = final_optim_state.additionalinfo.M
    F_opt = final_optim_state.f_x+final_optim_state.g_x

    return examine_f_direction_TruncatedNewton(pb, M_opt, x0)

    #
    ### Running algorithms
    #

    # optimizer = ProximalGradient()
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    # optimizer = ProximalGradient(extrapolation = AcceleratedProxGrad())
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    # optimdata[optimizer] = trace

    # optimizer = ProximalGradient(extrapolation = MFISTA(AcceleratedProxGrad()))
    # trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)

    # Alternating

    optparams = OptimizerParams(
        iterations_limit = 30,
        trace_length = 30,
    )

    optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton())
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace


    optimizer = PartlySmoothOptimizer(manifold_update = ManNewtonCG())
    trace = optimize!(pb, optimizer, x0, optparams=optparams, optimstate_extensions=StructuredSolvers.osext)
    optimdata[optimizer] = trace


    # # Constant manifold
    # x0 = project(M_opt, x0)
    # optimizer = PartlySmoothOptimizer(manifold_update = ManifoldTruncatedNewton(), update_selector=ConstantManifoldSelector(M_opt))
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
