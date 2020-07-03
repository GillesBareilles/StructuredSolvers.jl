
print_header(o::Optimizer) = error("print_header not implemented for optimizer $(typeof(o))")

update_fg∇f!(state::OptimizerState, pb) = error("update_fg∇f! not implemented for optimizer state $(typeof(state)) and problem $(typeof(pb)).")

update_iterate!(state, pb, optimizer) = error("update_iterate! not implemented for optimizer state $(typeof(state)), problem $(typeof(pb)) and optimizer $(typeof(optimizer)).")

display_logs(state, pb, optimizer, iteration) = error("display_logs not implemented for optimizer state $(typeof(state)), problem $(typeof(pb)) and optimizer $(typeof(optimizer)).")

display_logs_header(o::Optimizer) = error("display_logs_header not implemented for optimizer $(typeof(o))")


function optimize!(pb::CompositeProblem, optimizer::O, initial_x; state::S=initial_state(optimizer, initial_x)) where {O<:Optimizer, S<:OptimizerState}

    ## TODO: factor options
    show_trace = true
    iterations_limit = 20
    time_limit = 30
    tracing = show_trace
    stopped = false

    t0 = time()
    iteration = 0
    converged = false


    update_fg∇f!(state, pb)

    show_trace && print_header(optimizer)
    show_trace && display_logs_header(optimizer)
    show_trace && display_logs(state, pb, optimizer, iteration)

    while !converged && !stopped && iteration < iterations_limit
        iteration += 1

        @timeit to "iterate update" update_iterate!(state, pb, optimizer)

        @timeit to "oracles calls" update_fg∇f!(state, pb)

        show_trace && display_logs(state, pb, optimizer, iteration)

        _time = time()
        stopped_by_time_limit = _time-t0 > time_limit
        stopped = stopped_by_time_limit
    end

    return to
end
