
function print_header(o::Optimizer)
    return error("print_header not implemented for optimizer $(typeof(o))")
end

function update_fg∇f!(state::OptimizerState, pb)
    return error("update_fg∇f! not implemented for optimizer state $(typeof(state)) and problem $(typeof(pb)).")
end

function update_iterate!(state, pb, optimizer)
    return error("update_iterate! not implemented for optimizer state $(typeof(state)), problem $(typeof(pb)) and optimizer $(typeof(optimizer)).")
end

function display_logs(state, pb, optimizer, iteration, time, ose, tracing)
    return error("display_logs not implemented for optimizer state $(typeof(state)), problem $(typeof(pb)) and optimizer $(typeof(optimizer)).")
end

function display_logs_header(o::Optimizer)
    return error("display_logs_header not implemented for optimizer $(typeof(o))")
end


function optimize!(
    pb::CompositeProblem,
    optimizer::O,
    initial_x;
    state::S = initial_state(optimizer, initial_x, pb.regularizer),
    optimstate_extensions = [],
    iterations_limit = 30,
    show_trace = true,
    trace_length = 20,
) where {O<:Optimizer,S<:OptimizerState}

    ## TODO: factor options
    # show_trace = true
    # iterations_limit = 20
    time_limit = 30
    tracing = show_trace
    stopped = false

    t0 = time()
    iteration = 0
    converged = false


    update_fg∇f!(state, pb)

    show_trace && print_header(optimizer)
    show_trace && display_logs_header(optimizer)
    optimizationstate = display_logs(
        state,
        pb,
        optimizer,
        iteration,
        time() - t0,
        optimstate_extensions,
        tracing,
    )
    tr = Vector{OptimizationState}([optimizationstate])


    while !converged && !stopped && iteration < iterations_limit
        iteration += 1

        state.M_old = state.M
        state.x_old .= state.x


        @timeit to "iterate update" update_iterate!(state, pb, optimizer)

        @timeit to "oracles calls" update_fg∇f!(state, pb)


        _time = time()

        ## Decide if tracing this iteration
        tracing = show_trace && (mod(iteration, ceil(iterations_limit / trace_length)) == 0 || iteration==iterations_limit)

        optimizationstate = display_logs(
            state,
            pb,
            optimizer,
            iteration,
            _time - t0,
            optimstate_extensions,
            tracing,
        )
        push!(tr, optimizationstate)

        # converged = (norm(state.x - state.x_old) < 1e-7)
        # converged = (state.f_x + state.g_x < 1e-12)

        # stopped_by_empymanifold = manifold_dimension(state.M) == 0
        stopped_by_time_limit = _time - t0 > time_limit
        stopped = stopped_by_time_limit
    end

    return tr
end
