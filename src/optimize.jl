
function print_header(o::Optimizer)
    return error("print_header not implemented for optimizer $(typeof(o))")
end

function update_fg∇f!(state::OptimizerState, pb)
    return error("update_fg∇f! not implemented for optimizer state $(typeof(state)) and problem $(typeof(pb)).")
end

function update_iterate!(state, pb, optimizer)
    return error("update_iterate! not implemented for optimizer state $(typeof(state)), problem $(typeof(pb)) and optimizer $(typeof(optimizer)).")
end

function display_logs(state, pb, optimizer, iteration, time, ose)
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
    optimstate_extensions = [],
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

    if show_trace
        print_header(optimizer)
        display_logs_header(optimizer)
        optimizationstate = display_logs(state, pb, optimizer, iteration, time()-t0, optimstate_extensions)
        global tr = Vector([optimizationstate])

        @show typeof(tr)
    end

    trace = OptimizationState

    while !converged && !stopped && iteration < iterations_limit
        iteration += 1

        @timeit to "iterate update" update_iterate!(state, pb, optimizer)

        @timeit to "oracles calls" update_fg∇f!(state, pb)


        _time = time()
        if show_trace
            optimizationstate = display_logs(state, pb, optimizer, iteration, _time-t0, optimstate_extensions)
            push!(tr, optimizationstate)
        end

        stopped_by_time_limit = _time - t0 > time_limit
        stopped = stopped_by_time_limit
    end

    return to, tr
end
