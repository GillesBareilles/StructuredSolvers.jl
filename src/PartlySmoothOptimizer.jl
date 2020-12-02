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

mutable struct PartlySmoothOptimizerState{Tambiant, Tmanpoint, Tmanvec} <: OptimizerState
    it::Int64
    x::Point{Tambiant, Tmanpoint}
    x_old::Point{Tambiant, Tmanpoint}
    M::Manifold
    M_old::Manifold
    f_x::Float64
    g_x::Float64
    ∇f_x::Tambiant
    temppoint_amb::Tambiant
    tempvec_man::Tmanvec
    selected_update
    previous_update
    update_to_updatestate::Dict
end

function PartlySmoothOptimizerState(
    o::PartlySmoothOptimizer,
    x_amb::T,
    g::R,
    M::Manifold
) where {T,R}
    x_man = project(M, x_amb)
    return PartlySmoothOptimizerState(
        -1,
        Point(x_amb, x_man, M, ambiant_repr),
        Point(x_amb, x_man, M, ambiant_repr),
        M,
        M,
        -Inf,
        -Inf,
        zero(x_amb),
        zero(x_amb),
        zero_tangent_vector(M, x_man),
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
    println(" - update selection    ", summary(pso.update_selector))
    println(" - wholespace update   ", summary(pso.wholespace_update))
    println(" - manifold update     ", summary(pso.manifold_update))
end

function update_fg∇f!(state::PartlySmoothOptimizerState, pb)
    x = get_repr(state.x)

    state.f_x = f(pb, x)
    state.g_x = g(pb, x)
    ∇f!(pb, state.∇f_x, x)
    state.it += 1
    return
end

function initial_state(o::PartlySmoothOptimizer, x, reg; manifold=wholespace_manifold(reg, x))
    initstate = PartlySmoothOptimizerState(o, x, reg, manifold)
    initstate.update_to_updatestate[o.wholespace_update] = initial_state(o.wholespace_update, x, reg)
    initstate.update_to_updatestate[o.manifold_update] = initial_state(o.manifold_update, x, reg)
    return initstate
end




function convert_point_repr!(x::Point{Tamb, Tman}, ::WholespaceUpdate) where {Tamb, Tman}
    if x.repr == manifold_repr
        x.amb_repr = embed(x.M, x.man_repr)
        x.repr = ambiant_repr
    end
    return
end

function convert_point_repr!(x::Point{Tamb, Tman}, ::ManifoldUpdate) where {Tamb, Tman}
    if x.repr == ambiant_repr
        x.man_repr = project(x.M, x.amb_repr)
        x.repr = manifold_repr
    end
    return
end

function update_iterate!(ostate::PartlySmoothOptimizerState, pb, optimizer::PartlySmoothOptimizer)
    ostate.M = ostate.M_old

    select_update!(optimizer.update_selector, ostate, optimizer, pb)

    # A composite update expects points in ambiant space, while a manifold one in manifold representation.
    convert_point_repr!(ostate, optimizer, pb)

    update_iterate!(ostate, pb, ostate.selected_update)

    return
end



#
### Printing and logging
#
str_updatelog(::AbstractUpdate, ::OptimizerState) = ""
str_updatelog(::Nothing, ::OptimizerState) = ""

display_logs_header_post(o::PartlySmoothOptimizer) = print("Update")

function display_logs_pre(::PartlySmoothOptimizer, state, pb)
    return isa(state.selected_update,WholespaceUpdate) ? "90" : "0"
end

function display_logs_post(::PartlySmoothOptimizer, state, pb)
    @printf "%-13s\t" summary(state.selected_update)

    if !isnothing(state.selected_update)
        print(str_updatelog(state.selected_update, state.update_to_updatestate[state.selected_update]))
    end
    return
end
display_logs_header_post(::PartlySmoothOptimizerState) = print("      Update")
