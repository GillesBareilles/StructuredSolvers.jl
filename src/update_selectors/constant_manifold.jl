struct ConstantManifoldSelector{Tm} <: AbstractUpdateSelector
    manifold::Tm
end

function select_update!(constantsel::ConstantManifoldSelector, state, optimizer, pb)
    state.previous_update = state.selected_update
    state.selected_update = optimizer.manifold_update

    state.M = constantsel.manifold
    return
end

Base.summary(sel::ConstantManifoldSelector) = string("Cst ", sel.manifold)
