"""
    AlternatingUpdateSelector

Selects the composite update and manifold update every other iteration of the
`PartlySmoothOptimizer` method.
"""
struct AlternatingUpdateSelector <: AbstractUpdateSelector
end

function select_update!(::AlternatingUpdateSelector, state, optimizer, pb)
    state.previous_update = state.selected_update

    state.selected_update = optimizer.wholespace_update
    if state.previous_update == optimizer.wholespace_update && manifold_dimension(state.M)>0
        state.selected_update = optimizer.manifold_update
    end

    return
end

Base.summary(::AlternatingUpdateSelector) = "AltCompoMan"
