using CompositeProblems
using StructuredSolvers
using StructuredProximalOperators
using PGFPlotsX
using DataStructures
using Random

include("COAP_commonparameters.jl")
include("runexpenums.jl")

get_iterate(state) = state.x
get_manifold(state) = state.M

function get_problems()
    problems = Dict()

    seeds = 1

    ## Build regularizer
    n = 2
    p = 1.3
    regularizer = regularizer_distball(p=p, r=1.0, λ=1.0)

    m, sparsity = 3, 1
    Random.seed!(148)
    xstart = rand(n)
    xstart *= 0 / norm(xstart, p)

    problems["distball_randinit"] = []
    for seed in seeds
        pb = get_random_qualifiedleastsquares(n, m, regularizer, sparsity, seed = seed+4)
        push!(problems["distball_randinit"], (pb = pb, xstart = xstart))
    end

    return problems
end


function get_algorithms()
    algorithms = []

    # αuser = 0.001
    iterations_limit = 5e3
    printstep = 1e3

    optimstate_extens = [
        (key = :x, getvalue = get_iterate),
        (key = :M, getvalue = get_manifold)
    ]

    commonparams = Dict(
        :optimstate_extensions => optimstate_extens,
        :iterations_limit => iterations_limit,
        :show_trace => true
    )


    push!(algorithms, (
        name = "ISTA",
        optimizer = ProximalGradient(backtracking=true),
        params=commonparams
    ))

    p = 1/20
    apg_extrapolation = AcceleratedProxGrad(p=p,q=(p^2 + (2-p)^2)/2,r=4.0)

    push!(algorithms, (
        name = "T2",
        optimizer = ProximalGradient(backtracking=true, extrapolation = Test2ProxGrad(apg_extrapolation)),
        params=commonparams
    ))

    push!(algorithms, (
        name = "T1",
        optimizer = ProximalGradient(backtracking=true, extrapolation = Test1ProxGrad(apg_extrapolation)),
        params=commonparams
    ))

    push!(algorithms, (
        name = "FISTA",
        optimizer = ProximalGradient(backtracking=true, extrapolation = apg_extrapolation),
        params=commonparams
    ))

    return algorithms
end

FIGS_FOLDER = "./figs"
basename(pwd()) == "src" && (FIGS_FOLDER = joinpath("..", FIGS_FOLDER))
!ispath(FIGS_FOLDER) && mkpath(FIGS_FOLDER)


run_expenums(get_problems(), get_algorithms(), FIGS_FOLDER=FIGS_FOLDER)
