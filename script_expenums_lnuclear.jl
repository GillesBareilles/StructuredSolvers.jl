using CompositeProblems
using StructuredSolvers
using StructuredProximalOperators
using PGFPlotsX
using DataStructures
using Random
using Distributions

include("COAP_commonparameters.jl")
include("runexpenums.jl")

get_iterate(state) = state.x
get_manifold(state) = state.M

function get_problems()
    problems = Dict()

    seeds = 1:5

    ## Lasso l1
    problems = Dict()

    n = (7, 6)
    m = 2^2
    sparsity = 3

    # n = (10, 10)
    # m = 8^2
    # sparsity = 3

    Random.seed!(1234)
    xstart = rand(Normal(), n) .* 10

    problems["lnuclear_randinit"] = []
    for seed in seeds
        pb = get_random_qualifiedleastsquares(n, m, regularizer_lnuclear(1.0), sparsity, seed = seed+3)
        push!(problems["lnuclear_randinit"], (pb = pb, xstart = xstart))
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

FIGS_FOLDER = "./figs"
basename(pwd()) == "src" && (FIGS_FOLDER = joinpath("..", FIGS_FOLDER))
!ispath(FIGS_FOLDER) && mkpath(FIGS_FOLDER)


# run_expenums(get_problems(), get_algorithms(), FIGS_FOLDER=FIGS_FOLDER)
run_expenums_moyenne(get_problems(), get_algorithms(), FIGS_FOLDER=FIGS_FOLDER)
