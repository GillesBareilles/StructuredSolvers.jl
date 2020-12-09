struct ManifoldFollowingSelector <: AbstractUpdateSelector
end

function select_update!(::ManifoldFollowingSelector, state, optimizer, pb)
    state.previous_update = state.selected_update

    ## Use proximal gradient to detect comming manifold:
    γ = state.update_to_updatestate[optimizer.wholespace_update].γ

    state.temp .= state.x .- γ .* state.∇f_x
    M = prox_αg(pb, state.temp, γ)[2]
    state.ncalls_proxg += 1

    state.selected_update = optimizer.wholespace_update
    if state.M == M && manifold_dimension(M) > 0
        state.selected_update = optimizer.manifold_update
    end

    return
end

Base.summary(::ManifoldFollowingSelector) = "Adaptive"
