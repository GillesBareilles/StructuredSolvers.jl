using CompositeProblems
using StructuredSolvers
using StructuredProximalOperators
using PGFPlotsX
using DataStructures

include("COAP_commonparameters.jl")

get_iterate(state) = state.x


function main()
    n = 2
    pb = get_lasso(n, 12, 0.6)
    x0 = zeros(n) .+ 50


    ## Lasso 2d
    A = Matrix{Float64}([1.0 1.0; 0.0 1.0])
    y = Vector([1.0, 2.0])
    λ = 1.0

    pb = LassoPb(A, y, regularizer_l1(1.0), 2)

    objopt = 1.5
    xopt = [0.0, 1.0]

    x0 = Vector([4.0, 2.0]) + 50 * Vector([1.0, 1.0])

    γuser = 0.1


    optimstate_extens = [(key = :x, getvalue = get_iterate)]
    optimdata = OrderedDict{Optimizer,OptimizationTrace}()
    iterations_limit = 150


    optimizer = ProximalGradient(backtracking = false)
    init_state = ProximalGradientState(optimizer, x0, pb.regularizer; γ = γuser)
    to_pg, tr_pg = optimize!(
        pb,
        optimizer,
        x0;
        state = init_state,
        optimstate_extensions = optimstate_extens,
        iterations_limit = iterations_limit,
    )
    optimdata[optimizer] = tr_pg

    p = 1 / 20
    apg_extrapolation = AcceleratedProxGrad(p = p, q = (p^2 + (2 - p)^2) / 2, r = 4.0)
    optimizer = ProximalGradient(backtracking = false, extrapolation = apg_extrapolation)

    init_state = ProximalGradientState(optimizer, x0, pb.regularizer, γ = γuser)
    to_apg, tr_apg = optimize!(
        pb,
        optimizer,
        x0;
        state = init_state,
        optimstate_extensions = optimstate_extens,
        iterations_limit = iterations_limit,
    )
    optimdata[optimizer] = tr_apg


    optimizer = ProximalGradient(
        backtracking = false,
        extrapolation = Test1ProxGrad(apg_extrapolation),
    )
    init_state = ProximalGradientState(optimizer, x0, pb.regularizer, γ = γuser)
    to_t1, tr_t1 = optimize!(
        pb,
        optimizer,
        x0;
        state = init_state,
        optimstate_extensions = optimstate_extens,
        iterations_limit = iterations_limit,
    )
    optimdata[optimizer] = tr_t1

    optimizer = ProximalGradient(
        backtracking = false,
        extrapolation = Test2ProxGrad(apg_extrapolation),
    )
    init_state = ProximalGradientState(optimizer, x0, pb.regularizer, γ = γuser)
    to_t2, tr_t2 = optimize!(
        pb,
        optimizer,
        x0;
        state = init_state,
        optimstate_extensions = optimstate_extens,
        iterations_limit = iterations_limit,
    )
    optimdata[optimizer] = tr_t2

    optimizer =
        ProximalGradient(backtracking = false, extrapolation = MFISTA(apg_extrapolation))
    init_state = ProximalGradientState(optimizer, x0, pb.regularizer, γ = γuser)
    to_MAPG, tr_MAPG = optimize!(
        pb,
        optimizer,
        x0;
        state = init_state,
        optimstate_extensions = optimstate_extens,
        iterations_limit = iterations_limit,
    )
    optimdata[optimizer] = tr_MAPG


    plot_subopt = StructuredSolvers.plot_fvals_iteration(optimdata, objopt)
    pgfsave("subopt.tex", plot_subopt, include_preamble = true)
    pgfsave("subopt.pdf", plot_subopt)

    plot_it = StructuredSolvers.plot_iterates(pb, optimdata)
    pgfsave("iterates.tex", plot_it, include_preamble = true)
    pgfsave("iterates.pdf", plot_it)
    return
end


r = main()
