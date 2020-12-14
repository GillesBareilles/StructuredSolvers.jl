###########################################################
### Proximal Gradient
###########################################################
@with_kw struct WholespaceProximalGradient <: WholespaceUpdate
    backtracking::Bool = true
    γ_init = 1e5
end
@with_kw mutable struct WholespaceProximalGradientState <: AbstractUpdateState
    γ::Float64
end

Base.summary(::WholespaceProximalGradient) = "ProxGrad"

initial_state(o::WholespaceProximalGradient, x, reg; manifold=nothing) = WholespaceProximalGradientState(γ = o.γ_init)

str_updatelog(o::WholespaceProximalGradient, t::WholespaceProximalGradientState) = @sprintf "γ: %.3e" t.γ

function update_iterate!(state::PartlySmoothOptimizerState, pb, m::WholespaceProximalGradient)
    γ = state.update_to_updatestate[m].γ

    @assert state.x.repr == ambiant_repr

    if m.backtracking
        ncalls_f, γ = backtrack_f_lipschitzgradient!(state, pb, γ)
    end

    state.temppoint_amb .= state.x.amb_repr .- γ .* state.∇f_x

    state.x.man_repr, M = prox_αg(pb, state.temppoint_amb, γ); state.ncalls_proxg += 1
    state.x.repr = manifold_repr
    state.x.M = M

    state.M = M
    state.update_to_updatestate[m].γ = γ

    return
end


function display_logs()
    # γ          %.3e  state.γ
end






function backtrack_f_lipschitzgradient!(state::PartlySmoothOptimizerState, pb, γ)
    τ = 1.2

    ∇f_norm2 = norm(state.∇f_x)^2

    itmax = 200
    ncalls_f = 0

    it_ls = 0
    while it_ls <= itmax
        state.temppoint_amb .= state.x.amb_repr .- γ .* state.∇f_x

        # f(pb, state.temp) ≤ state.f_x - 1/(2*γ) * norm(state.temp-state.x)^2 && break
        state.ncalls_f += 1
        f(pb, state.temppoint_amb) ≤ state.f_x - γ / 2 * ∇f_norm2 && break

        ncalls_f += 1
        γ = γ / τ
        it_ls += 1
    end

    if it_ls > itmax
        @warn "Gradient backtracking: reached iterations limits."
    end


    return ncalls_f, γ
end
