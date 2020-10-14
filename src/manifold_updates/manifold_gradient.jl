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

str_updatelog(o::ManifoldGradient, t::ManifoldGradientState) = @sprintf "ls-nit %2i" t.ls_niter

function update_iterate!(state::PartlySmoothOptimizerState, pb, o::ManifoldGradient)
    grad_fgₖ = state.tempvec_man
    M = state.M
    x = state.x.man_repr

    @assert state.x.repr == manifold_repr
    @assert is_manifold_point(M, x)

    # TODO: remove intermediate alloc from .+= op.
    grad_fgₖ = egrad_to_rgrad(state.M, x, state.∇f_x) + ∇M_g(pb, state.M, x)

    state.update_to_updatestate[o].norm_∇fgₘ = norm(M, x, grad_fgₖ)

    # TODO: make linesearch inplace for x, return status.
    hist_ls = Dict()
    x_ls = linesearch(o.linesearch, pb, state.M, state.x.man_repr, grad_fgₖ, -grad_fgₖ, hist=hist_ls)

    state.x.man_repr = x_ls
    state.update_to_updatestate[o].ls_niter = hist_ls[:niter]
    return
end
