struct TwoPhaseTargetSelector <: AbstractUpdateSelector
    M_target::Manifold
end

function select_update!(selector::TwoPhaseTargetSelector, state, optimizer, pb)
    state.previous_update = state.selected_update

    if state.M == selector.M_target
        state.selected_update = optimizer.manifold_update
    else
        state.selected_update = optimizer.wholespace_update
    end

    return
end

Base.summary(::TwoPhaseTargetSelector) = "TwoPhaseTarget"
