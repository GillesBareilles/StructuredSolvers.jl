using CompositeProblems
using StructuredSolvers
using StructuredProximalOperators
using PGFPlotsX
using DataStructures
using Random
using Distributions

include("COAP_commonparameters.jl")
include("execution.jl")
include("plotting.jl")


get_iterate(state) = state.x
get_manifold(state) = state.M

function get_problems()
    problems = Dict()

    seeds = 1:30

    ## Nuclear least squares
    problems = []

    n = (7, 6)
    m = 2^2
    sparsity = 3

    for seed in seeds
        Random.seed!(seed+150)
        xstart = rand(Normal(), n) .* 10

        pb = get_random_qualifiedleastsquares(n, m, regularizer_lnuclear(1.0), sparsity, seed = seed)
        push!(problems, (pb = pb, xstart = xstart))
    end

    return problems
end


function get_algorithms()
    algorithms = []

    # αuser = 0.001
    iterations_limit = 1e4
    printstep = 1e3

    optimstate_extens = [
        (key = :x, getvalue = get_iterate),
        (key = :M, getvalue = get_manifold)
    ]

    commonparams = Dict(
        :optimstate_extensions => optimstate_extens,
        :iterations_limit => iterations_limit,
        :show_trace => false
    )


    push!(algorithms, (
        name = "ISTA",
        optimizer = ProximalGradient(backtracking=true),
        params=commonparams
    ))

    p = 1/20
    apg_extrapolation = AcceleratedProxGrad(p=p,q=(p^2 + (2-p)^2)/2,r=4.0)

    push!(algorithms, (
        name = "FISTA",
        optimizer = ProximalGradient(backtracking=true, extrapolation = apg_extrapolation),
        params=commonparams
    ))

    push!(algorithms, (
        name = "T1",
        optimizer = ProximalGradient(backtracking=true, extrapolation = Test1ProxGrad(apg_extrapolation)),
        params=commonparams
    ))

    push!(algorithms, (
        name = "T2",
        optimizer = ProximalGradient(backtracking=true, extrapolation = Test2ProxGrad(apg_extrapolation)),
        params=commonparams
    ))

    return algorithms
end


function main()
    FIGS_FOLDER = "./figs"
    basename(pwd()) == "src" && (FIGS_FOLDER = joinpath("..", FIGS_FOLDER))
    !ispath(FIGS_FOLDER) && mkpath(FIGS_FOLDER)


    ## Execute all algorithms
    algo_pb_trace = run_algorithms_on_problems(get_problems(), get_algorithms())

    ## Build performance indicator per algo, problem
    get_abscisses(alg, pb, trace) = [state.it for state in trace]
    function get_ordinates(alg, pb, trace)
        M_x0 = pb.pb.M_x0
        Ms = [state.additionalinfo.M for state in trace]
        return 100*get_proportion_identifiedstructure(Ms, M_x0)
    end
    algo_pb_curve = extract_curves(algo_pb_trace, get_abscisses, get_ordinates)


    # plot_all_curves(
    #     algo_pb_curve,
    #     "reg_lnuclear_randinit_allcurves",
    #     FIGS_FOLDER = "./figs",
    # )

    # plot_aggregateproblems(
    #     algo_pb_curve,
    #     aggregate_pb_perfs = x -> quantile(x, 0.5),
    #     "reg_lnuclear_randinit_median",
    #     FIGS_FOLDER = "./figs",
    # )

    plot_aggregateproblems(
        algo_pb_curve,
        aggregate_pb_perfs = x -> mean(x),
        "reg_lnuclear_randinit_mean",
        FIGS_FOLDER = "./figs",
    )

    # plot_aggregateproblems_fillbetween(algo_pb_curve, "reg_lnuclear_fillbetween", FIGS_FOLDER="./figs")

    # plot_performance_profile(algo_pb_trace, )

    return
end

main()
