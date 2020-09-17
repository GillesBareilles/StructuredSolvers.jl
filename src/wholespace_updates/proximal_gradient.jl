###########################################################
### Proximal Gradient
###########################################################
@with_kw struct WholespaceProximalGradient <: WholespaceUpdate
    backtracking::Bool = true
end
@with_kw mutable struct WholespaceProximalGradientState <: AbstractUpdateState
    γ::Float64 = 1e5
end

Base.summary(::WholespaceProximalGradient) = "ProxGrad"

initial_state(::WholespaceProximalGradient, x, reg) = WholespaceProximalGradientState()

str_updatelog(o::WholespaceProximalGradient, t::WholespaceProximalGradientState) = @sprintf "γ: %.3e" t.γ

function update_iterate!(state::PartlySmoothOptimizerState, pb, m::WholespaceProximalGradient)
    γ = state.update_to_updatestate[m].γ
    if m.backtracking
        ncalls_f, γ = backtrack_f_lipschitzgradient!(state, pb, γ)
    end

    state.temp .= state.x .- γ .* state.∇f_x
    M = prox_αg!(pb, state.x, state.temp, γ)
    state.M = M
    state.update_to_updatestate[m].γ = γ

    return
end


function display_logs()
    # γ          %.3e  state.γ
end
