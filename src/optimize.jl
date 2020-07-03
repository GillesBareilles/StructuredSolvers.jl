
function print_header(o::Optimizer)
    return error("print_header not implemented for optimizer $(typeof(o))")
end

function update_fg∇f!(state::OptimizerState, pb)
    return error("update_fg∇f! not implemented for optimizer state $(typeof(state)) and problem $(typeof(pb)).")
end

function update_iterate!(state, pb, optimizer)
    return error("update_iterate! not implemented for optimizer state $(typeof(state)), problem $(typeof(pb)) and optimizer $(typeof(optimizer)).")
end

function display_logs(state, pb, optimizer, iteration)
    return error("display_logs not implemented for optimizer state $(typeof(state)), problem $(typeof(pb)) and optimizer $(typeof(optimizer)).")
end

function display_logs_header(o::Optimizer)
    return error("display_logs_header not implemented for optimizer $(typeof(o))")
end


function optimize!(
    pb::CompositeProblem,
    optimizer::O,
    initial_x;
    state::S = initial_state(optimizer, initial_x),
) where {O<:Optimizer,S<:OptimizerState}

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
        stopped_by_time_limit = _time - t0 > time_limit
        stopped = stopped_by_time_limit
    end

    return to
end
