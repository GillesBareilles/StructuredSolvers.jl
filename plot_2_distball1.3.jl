using CompositeProblems
using StructuredSolvers
using StructuredProximalOperators
using PGFPlotsX
using DataStructures

include("COAP_commonparameters.jl")

function main()
    #
    ## Problem setup
    #
    θ = π*(0.5 + 0.2)
    x0 = 3 .* [cos(θ), sin(θ)]

    p_norm = 1.3

    xopt = [cos(θ), sin(θ)]; xopt /= norm(xopt, p_norm)
    objopt = 0.

    A = Matrix{Float64}([1. 1.; 0. 1.])
    y = A * xopt
    λ = 1.0

    pb = LeastsquaresPb(A, y, regularizer_distball(1.0, p_norm, λ), 2)

    γuser = 0.05

    #
    ## Execution of algorithms
    #
    optimstate_extens = [(key = :x, getvalue = get_iterate)]
    optimdata = OrderedDict{Optimizer,OptimizationTrace}()
    iterations_limit = 150


    optimizer = ProximalGradient(backtracking=false)
    init_state = ProximalGradientState(optimizer, x0, pb.regularizer; γ = γuser)
    to_pg, tr_pg = optimize!(pb, optimizer, x0; state=init_state, optimstate_extensions = optimstate_extens, iterations_limit=iterations_limit)
    optimdata[optimizer] = tr_pg

    p = 1/20
    apg_extrapolation = AcceleratedProxGrad(p=p,q=(p^2 + (2-p)^2)/2,r=4.0)
    optimizer = ProximalGradient(backtracking=false, extrapolation = apg_extrapolation)

    init_state = ProximalGradientState(optimizer, x0, pb.regularizer, γ = γuser)
    to_apg, tr_apg = optimize!(pb, optimizer, x0; state=init_state, optimstate_extensions = optimstate_extens, iterations_limit=iterations_limit)
    optimdata[optimizer] = tr_apg


    optimizer = ProximalGradient(backtracking=false, extrapolation = Test1ProxGrad(apg_extrapolation))
    init_state = ProximalGradientState(optimizer, x0, pb.regularizer, γ = γuser)
    to_t1, tr_t1 = optimize!(pb, optimizer, x0; state=init_state, optimstate_extensions = optimstate_extens, iterations_limit=iterations_limit)
    optimdata[optimizer] = tr_t1

    optimizer = ProximalGradient(backtracking=false, extrapolation = Test2ProxGrad(apg_extrapolation))
    init_state = ProximalGradientState(optimizer, x0, pb.regularizer, γ = γuser)
    to_t2, tr_t2 = optimize!(pb, optimizer, x0; state=init_state, optimstate_extensions = optimstate_extens, iterations_limit=iterations_limit)
    optimdata[optimizer] = tr_t2

    optimizer = ProximalGradient(backtracking=false, extrapolation = MFISTA(apg_extrapolation))
    init_state = ProximalGradientState(optimizer, x0, pb.regularizer, γ = γuser)
    to_MAPG, tr_MAPG = optimize!(pb, optimizer, x0; state=init_state, optimstate_extensions = optimstate_extens, iterations_limit=iterations_limit)
    optimdata[optimizer] = tr_MAPG


    plot_subopt = StructuredSolvers.plot_fvals_iteration(optimdata, objopt)
    pgfsave("distball_$(p_norm)_subopt.tex", plot_subopt, include_preamble = true)
    pgfsave("distball_$(p_norm)_subopt.pdf", plot_subopt)

    plot_it = StructuredSolvers.plot_iterates(pb, optimdata)
    pgfsave("distball_$(p_norm)_iterates.tex", plot_it, include_preamble = true)
    pgfsave("distball_$(p_norm)_iterates.pdf", plot_it)
    return
end


r = main()
