
function print_header(o::Optimizer)
    return error("print_header not implemented for optimizer $(typeof(o))")
end

function update_fg∇f!(state::OptimizerState, pb)
    return error("update_fg∇f! not implemented for optimizer state $(typeof(state)) and problem $(typeof(pb)).")
end

function update_iterate!(state, pb, optimizer)
    return error("update_iterate! not implemented for optimizer state $(typeof(state)), problem $(typeof(pb)) and optimizer $(typeof(optimizer)).")
end

#
## Print and logs
#

display_logs_header_pre(o) = nothing
display_logs_header_post(o) = nothing

function display_logs_header(o::Optimizer)
    display_logs_header_pre(o::Optimizer)
    print("it.   F(x)                    f(x)       g(x)       step         tgt ∇f+g   nml ∇f+g     Manifold                          ")
    display_logs_header_post(o::Optimizer)
    println()
    return
end


display_logs_pre(optimizer, state, pb) = "0"
display_logs_post(optimizer, state, pb) = ""

function display_logs(state, pb, optimizer, iteration, time, optimstate_extensions, tracing)
    x = get_repr(state.x)
    x_old = get_repr(state.x_old)

    normstep = norm(embed(state.x) - embed(state.x_old))
    minsubgradient_tan, minsubgradient_norm = firstorder_optimality_tangnorm(pb, get_repr(state.x), state.M, state.∇f_x)

    if tracing
        F_x = state.f_x + state.g_x

        linestyle = display_logs_pre(optimizer, state, pb)
        print("\033[$(linestyle)m")

        F_x = state.f_x + state.g_x
        @printf "%4i  %.16e  %-.3e  %-.3e  %-.3e    %-.3e  % .3e " iteration F_x state.f_x state.g_x normstep minsubgradient_tan minsubgradient_norm

        ## underlined manifold if state changed
        manstyle = (state.M == state.M_old) ? linestyle : "4"
        if length(string(state.M)) <= 35
            print(repeat(" ", 35-length(string(state.M))), "\033[$(manstyle)m",state.M, "\033[0m\033[$(linestyle)m")
        else
            print("\033[$(manstyle)m",state.M, "\033[0m\033[$(linestyle)m")
        end
        print("  ")

        display_logs_post(optimizer, state, pb)

        print("\033[0m")
        println()
    end

    return build_optimstate(optimizer, state, iteration, time, normstep, minsubgradient_tan, minsubgradient_norm, optimstate_extensions)
end





function optimize!(
    pb::CompositeProblem,
    optimizer::O,
    initial_x;
    manifold = wholespace_manifold(pb.regularizer, initial_x),
    state::S = initial_state(optimizer, initial_x, pb.regularizer, manifold=manifold),
    optimstate_extensions = [],
    optparams = OptimizerParams()
) where {O<:Optimizer,S<:OptimizerState}

    if getfield(StructuredSolvers, :timeit_debug_enabled)()
        reset_timer!()
    end

    ## Collecting parameters
    @unpack iterations_limit, time_limit = optparams
    @unpack show_trace, trace_length = optparams
    tracing = show_trace

    iteration = 0
    converged = false
    stopped = false
    time_count = 0.0


    update_fg∇f!(state, pb)

    show_trace && print_header(optimizer)
    show_trace && display_logs_header(optimizer)
    optimizationstate = display_logs(
        state,
        pb,
        optimizer,
        iteration,
        time_count,
        optimstate_extensions,
        tracing,
    )
    tr = Vector{OptimizationState}([optimizationstate])


    while !converged && !stopped && iteration < iterations_limit
        iteration += 1

        state.M_old = state.M
        state.x_old = deepcopy(state.x)


        _time = time()
        @timeit_debug "update_iterate" update_iterate!(state, pb, optimizer)

        @timeit_debug "oracles calls" update_fg∇f!(state, pb)
        time_count += time() - _time



        ## Decide if tracing this iteration
        tracing = show_trace && (mod(iteration, ceil(iterations_limit / trace_length)) == 0 || iteration==iterations_limit)

        optimizationstate = display_logs(
            state,
            pb,
            optimizer,
            iteration,
            time_count,
            optimstate_extensions,
            tracing,
        )
        push!(tr, optimizationstate)

        converged = false
        for cvchecker in optparams.cvcheckers
            converged = converged || has_converged(cvchecker, pb, optimizer, optimizationstate)
        end

        # stopped_by_empymanifold = manifold_dimension(state.M) == 0
        stopped_by_time_limit = time_count > time_limit
        stopped = stopped_by_time_limit
    end

    if show_trace && !tracing
        optimizationstate = display_logs(
            state,
            pb,
            optimizer,
            iteration,
            time_count,
            optimstate_extensions,
            true,
        )
        push!(tr, optimizationstate)
    end

    ## Optimality of last iterate
    x = state.x
    M = state.M

    if show_trace
        println("Optimality status of last iterate:")
        res_tan, res_norm = firstorder_optimality_tangnorm(pb, get_repr(state.x), M, state.∇f_x)

        println("- ||Π_tangent(∇f+ḡ)||      : ", res_tan)
        println("- ||Π_normal(∇f+ḡ)||       : ", res_norm)
    end


    if getfield(StructuredSolvers, :timeit_debug_enabled)()
        printstyled("\n\n")
        printstyled("**********************************\n", color=:red)
        print_timer()
        printstyled("**********************************", color=:red)
        printstyled("\n\n")
    end



    return tr
end
