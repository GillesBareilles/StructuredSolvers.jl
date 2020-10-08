function has_converged(cvchecker::ConvergenceChecker, optimizerstate::OptimizerState)
    return false
end


@with_kw struct ProxGradStepLength <: ConvergenceChecker
    tol::Float64 = 1e-15
end

function has_converged(cvchecker::ProxGradStepLength, pb::CompositeProblem, optimizer::Optimizer, state::PartlySmoothOptimizerState)
    γ = state.update_to_updatestate[optimizer.wholespace_update].γ

    # state.temp .= state.x .- γ .* state.∇f_x
    # res, M = prox_αg(pb, state.temp, γ)

    # @show norm((state.x - res) / γ)
    # return norm(state.x - state.temp) < cvchecker.tol
    return false
end

function has_converged(cvchecker::ProxGradStepLength, pb::CompositeProblem, o::Optimizer, state::ProximalGradientState)
    γ = state.γ
    # state.temp .= state.x .- γ .* state.∇f_x
    # return norm(state.x - prox_αg(pb, state.temp, γ)[1]) < cvchecker.tol
    return false
end
