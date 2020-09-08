"""
    run_algorithms_on_problems(problems, algorithms)

Run all `algorithms` on all `problems` and returns a collection
`algorithm × problem → trace::Vector{OptimizationState}`
"""
function run_algorithms_on_problems(problems, algorithms)
    algo_pb_trace = OrderedDict()
    for algo in algorithms
        printstyled("---- Algorithm $(algo.name)\n", color = :light_blue)

        pb_trace = Dict()
        for pb in problems
            printstyled("----- Solving problem...\n", color = :light_blue)

            pb_trace[pb] = optimize!(pb.pb, algo.optimizer, pb.xstart; algo.params...)
        end
        algo_pb_trace[algo] = pb_trace
    end
    return algo_pb_trace
end


"""
    extract_curves(algo_pb_curve, get_abscisses, get_ordinates)

Extracts a collection `algo_pb_curve`: `algorithm × problem → curve = (abscisses, ordinates)`
from `algo_pb_curve`: `algorithm × problem → trace::Vector{OptimizationState}` using
`get_abscisses`, `get_ordinates`.
"""
function extract_curves(algo_pb_trace, get_abscisses, get_ordinates)
    algo_pb_curve = OrderedDict()
    for (algo, pb_trace) in algo_pb_trace
        pb_curve = OrderedDict()
        for (pb_id, pbtrace) in enumerate(pb_trace)
            pb, trace = pbtrace

            abscisses = get_abscisses(algo, pb, trace)
            ordinates = get_ordinates(algo, pb, trace)

            pb_curve[pb] = (abscisses = abscisses, ordinates = ordinates)
        end
        algo_pb_curve[algo] = pb_curve
    end
    return algo_pb_curve
end
