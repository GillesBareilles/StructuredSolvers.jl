###########################################################
### Manifold Gradient
###########################################################
@with_kw struct ManifoldGradient <: ManifoldUpdate
    linesearch::ManifoldLinesearch = ArmijoGoldstein()
end

@with_kw mutable struct ManifoldGradientState <: AbstractUpdateState
    norm_∇fgₘ::Float64 = -1.0
    ls_niter::Int64 = -1
end

initial_state(::ManifoldGradient, x, reg) = ManifoldGradientState()

Base.summary(o::ManifoldGradient) = "ManGradient"

str_updatelog(o::ManifoldGradient, t::ManifoldGradientState) = @sprintf "|∇(f+g)ₘ|: %.2e   ls-nit%2i" t.norm_∇fgₘ t.ls_niter

function update_iterate!(state::PartlySmoothOptimizerState, pb, o::ManifoldGradient)
    @assert is_manifold_point(state.M, state.x)

    grad_fgₖ = @view state.temp[:]     # makes code more readable

    grad_fgₖ .= egrad_to_rgrad(state.M, state.x, state.∇f_x) + ∇M_g(pb, state.M, state.x)

    hist_ls = Dict()
    x_ls = linesearch(o.linesearch, pb, state.M, state.x, grad_fgₖ, -grad_fgₖ, hist=hist_ls)

    state.x .= x_ls
    state.update_to_updatestate[o].norm_∇fgₘ = norm(state.M, state.x, grad_fgₖ)
    state.update_to_updatestate[o].ls_niter = hist_ls[:niter]
    return
end
