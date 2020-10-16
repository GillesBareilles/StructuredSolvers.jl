###########################################################
### Manifold Gradient
###########################################################
@with_kw struct ManifoldGradient <: ManifoldUpdate
    linesearch::ManifoldLinesearch = ArmijoGoldstein()
end
Base.summary(o::ManifoldGradient) = "ManGradient"

@with_kw mutable struct ManifoldGradientState <: AbstractUpdateState
    norm_∇fgₘ::Float64 = -1.0
    ls_niter::Int64 = -1
end

initial_state(::ManifoldGradient, x, reg) = ManifoldGradientState()


str_updatelog(o::ManifoldGradient, t::ManifoldGradientState) = @sprintf "ls-nit %2i\t\t||gradₘ f+g|| %.3e" t.ls_niter t.norm_∇fgₘ

function update_iterate!(state::PartlySmoothOptimizerState, pb, o::ManifoldGradient)
    grad_fgₖ = state.tempvec_man
    M = state.M
    x = state.x.man_repr

    @assert state.x.repr == manifold_repr
    @assert is_manifold_point(M, x)

    # TODO: remove intermediate alloc from .+= op.
    grad_fgₖ = egrad_to_rgrad(M, x, state.∇f_x) + ∇M_g(pb, M, x)

    state.update_to_updatestate[o].norm_∇fgₘ = norm(M, x, grad_fgₖ)

    # TODO: make linesearch inplace for x, return status.
    hist_ls = Dict()
    x_ls = linesearch(o.linesearch, pb, M, x, grad_fgₖ, -grad_fgₖ, hist=hist_ls)

    state.x.man_repr = x_ls
    state.update_to_updatestate[o].ls_niter = hist_ls[:niter]
    return
end
