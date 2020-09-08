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
    problems = []

    nseeds = 1

    ## Build regularizer
    ngroups = 10
    groups = [(5 * i - 4):(5 * i) for i in 1:ngroups]
    n = sum(length(group) for group in groups)
    reg_db = regularizer_distball(p = 2.0, r = 1.0, λ = 1.0)

    regularizer = regularizer_group(groups = groups, regs = [reg_db for i in 1:ngroups])


    m, sparsity = 40, 0
    Random.seed!(1236)
    xstart = rand(n) .* 10

    for seed in 1:nseeds
        Random.seed!(seed)
        η = 2
        A = rand(Normal(), m, n)
        for i in 1:min(m, n)
            A[i, i] += η
        end

        pb = get_random_qualifiedleastsquares(n, m, regularizer, sparsity, seed = seed+1, A = A)
        push!(problems, (pb = pb, xstart = xstart))
    end


    # ## Lasso l1 zero init
    # l1_zeroinit_pbs = []

    # for pb in l1_pbs
    #     push!(l1_zeroinit_pbs, (name = "pblasso_l1", pb = pb.pb, xstart = zeros(n)))
    # end

    # problems["l1_zeroinit"] = l1_zeroinit_pbs

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


    plot_all_curves(
        algo_pb_curve,
        "reg_groupdistball",
        FIGS_FOLDER = "./figs",
        xmode = "log"
    )

    # plot_aggregateproblems(
    #     algo_pb_curve,
    #     aggregate_pb_perfs = x -> quantile(x, 0.5),
    #     "reg_lnuclear_randinit_median",
    #     FIGS_FOLDER = "./figs",
    # )

    # plot_aggregateproblems(
    #     algo_pb_curve,
    #     aggregate_pb_perfs = x -> mean(x),
    #     "reg_lnuclear_randinit_mean",
    #     FIGS_FOLDER = "./figs",
    # )

    # plot_aggregateproblems_fillbetween(algo_pb_curve, "reg_lnuclear_fillbetween", FIGS_FOLDER="./figs")

    # plot_performance_profile(algo_pb_trace, )

    return
end

main()
