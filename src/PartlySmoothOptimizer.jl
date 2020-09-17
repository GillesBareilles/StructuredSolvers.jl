abstract type AbstractUpdateSelector end
abstract type AbstractUpdate end
abstract type WholespaceUpdate <: AbstractUpdate end
abstract type ManifoldUpdate <: AbstractUpdate end
abstract type AbstractUpdateState end


@with_kw struct PartlySmoothOptimizer{UpSel, WhUp, ManUp} <: Optimizer
    update_selector::UpSel = AlternatingUpdateSelector()
    wholespace_update::WhUp = WholespaceProximalGradient()
    manifold_update::ManUp = ManifoldIdentity()
end

mutable struct PartlySmoothOptimizerState{Tx} <: OptimizerState
    it::Int64
    x::Tx
    x_old::Tx
    M::Manifold
    M_old::Manifold
    f_x::Float64
    g_x::Float64
    ∇f_x::Tx
    temp::Tx
    selected_update
    previous_update
    update_to_updatestate::Dict
end
function PartlySmoothOptimizerState(
    o::PartlySmoothOptimizer,
    x::Tx,
    g::R;
) where {Tx,R}
    return PartlySmoothOptimizerState(
        -1,
        copy(x),
        copy(x),
        wholespace_manifold(g, x),
        wholespace_manifold(g, x),
        0.0,
        0.0,
        zero(x),
        zero(x),
        nothing,
        nothing,
        Dict()
    )
end

Base.summary(o::PartlySmoothOptimizer) = string(
    "PSOpt - ", summary(o.update_selector), " - ",
    summary(o.wholespace_update), " - ", summary(o.manifold_update)
)

function print_header(pso::PartlySmoothOptimizer)
    println("---------------------")
    println("--- Partly Smooth Optimizer")
    println(" - update selection    ", pso.update_selector)
    println(" - wholespace update   ", pso.wholespace_update)
    println(" - manifold update     ", pso.manifold_update)
end

function update_fg∇f!(state::PartlySmoothOptimizerState, pb)
    state.f_x = f(pb, state.x)
    state.g_x = g(pb, state.x)
    ∇f!(pb, state.∇f_x, state.x)
    state.it += 1
    return
end

function initial_state(o::PartlySmoothOptimizer, x, reg)
    initstate = PartlySmoothOptimizerState(o, x, reg)
    initstate.update_to_updatestate[o.wholespace_update] = initial_state(o.wholespace_update, x, reg)
    initstate.update_to_updatestate[o.manifold_update] = initial_state(o.manifold_update, x, reg)
    return initstate
end

function update_iterate!(state::PartlySmoothOptimizerState, pb, optimizer::PartlySmoothOptimizer)
    state.M = state.M_old
    select_update!(optimizer.update_selector, state, optimizer, pb)

    update_iterate!(state, pb, state.selected_update)
    return
end



function display_logs_header(o::PartlySmoothOptimizer)
    print("it.   F(x)                    f(x)       g(x)       step         Manifold      Update\n")
    return
end

str_updatelog(::AbstractUpdate, ::OptimizerState) = ""
str_updatelog(::Nothing, ::OptimizerState) = ""

function display_logs(
    state::PartlySmoothOptimizerState,
    pb,
    optimizer,
    iteration,
    time,
    optimstate_extensions,
    tracing,
)
    if tracing
        F_x = state.f_x + state.g_x
        @printf "%4i  %.16e  %-.3e  %-.3e  %-.3e  " iteration F_x state.f_x state.g_x norm(state.x - state.x_old)
        if state.M == state.M_old
            @printf "  %12s" state.M
        else

            @printf "%s\033[4m%s\033[0m" repeat(" ", 2+12-length(string(state.M))) state.M
        end
        @printf "  %-10s\t" summary(state.selected_update)

        if !isnothing(state.selected_update)
            printstyled(str_updatelog(state.selected_update, state.update_to_updatestate[state.selected_update]))
        end
        println()
    end

    ## TODO: This section of code is generic, should be factored. Plus, can the many allocs be avoided when zipping and building temp arrays?
    keys = []
    values = []
    for osextension in optimstate_extensions
        push!(keys)
        push!(values, osextension.getvalue(state))
    end

    return OptimizationState(
        it = iteration,
        time = time,
        f_x = state.f_x,
        g_x = state.g_x,
        norm_step = norm(state.x-state.x_old),
        additionalinfo = (;
            zip(
                [osextension.key for osextension in optimstate_extensions],
                [
                    copy(osextension.getvalue(state))
                    for osextension in optimstate_extensions
                ],
            )...),
    )
end
