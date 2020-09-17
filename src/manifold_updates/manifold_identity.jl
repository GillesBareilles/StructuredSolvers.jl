###########################################################
### Manifold Gradient
###########################################################
struct ManifoldIdentity <: ManifoldUpdate
end

struct ManifoldIdentityState
end

initial_state(::ManifoldIdentity, x, reg) = ManifoldIdentityState()

str_updatelog(::ManifoldIdentity, ::ManifoldIdentityState) = ""

function update_iterate!(state::PartlySmoothOptimizerState, pb, ::ManifoldIdentity)
    return
end
