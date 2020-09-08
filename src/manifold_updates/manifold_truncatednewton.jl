###########################################################
### Manifold Truncated Newton
###########################################################
"""
    ManifoldTruncatedNewton

Optimizer performing inexact Newton steps on a prescribed manifold. The inexactness in
the approximation of the Newton step is controlled by `νₖ`: one expects a direction `dᴺ`
such that
        ||∇²(f+g)(x)[dᴺ] + ∇(f+g)|| ≤ νₖ ||∇(f+g)||
"""
@with_kw struct ManifoldTruncatedNewton <: ManifoldUpdate
    linesearch::ManifoldLinesearch = ArmijoGoldstein()
    ν_reductionfactor::Float64 = 0.5
end
Base.summary(::ManifoldTruncatedNewton) = "ManTruncNewton"

@with_kw mutable struct ManifoldTruncatedNewtonState <: AbstractUpdateState
    norm_∇fgₘ::Float64 = -1.0
    νₖ::Float64 = 1.0
    CG_niter::Int64 = -1
    CG_ε::Float64 = -1
    CG_residual::Float64 = -1
    norm_dᴺ::Float64 = -1
    cosθ::Float64 = -1
    λ_x::Float64 = -1
    ls_niter::Int64 = -1
end

initial_state(::ManifoldTruncatedNewton, x, reg) = ManifoldTruncatedNewtonState()

str_updatelog(::ManifoldTruncatedNewton, t::ManifoldTruncatedNewtonState) = @sprintf "|∇(f+g)ₘ|: %.2e   CG: nit:%3i  νₖ:%.1e ε:%.1e residual:%.1e     |dᴺ|:%.3e   cos(θₖ):%.1e  λ(x):%.1e   ls: nit:%2i" t.norm_∇fgₘ t.CG_niter t.νₖ t.CG_ε t.CG_residual t.norm_dᴺ t.cosθ t.λ_x t.ls_niter

function update_iterate!(state::PartlySmoothOptimizerState, pb, o::ManifoldTruncatedNewton)

    man_truncnewton_state = state.update_to_updatestate[o]
    state.x_old .= state.x
    grad_fgₖ = @view state.temp[:]     # makes code more readable

    ncalls_f = 0
    ncalls_∇f = 0
    ncalls_∇²fh = 0

    @unpack ν_reductionfactor = o
    @unpack νₖ = man_truncnewton_state

    ## 1. Get first & second order information
    grad_fgₖ .= egrad_to_rgrad(state.M, state.x, state.∇f_x) + ∇M_g(pb, state.M, state.x)
    norm_rgrad = norm(state.M, state.x, grad_fgₖ)

    function hessfg_x_h(ξ)
        ncalls_∇²fh += 1
        return ehess_to_rhess(state.M, state.x, state.∇f_x, ∇²f_h(pb, state.x, ξ), ξ) + ∇²M_g_ξ(pb, state.M, state.x, ξ)
    end

    ncalls_∇f += 1
    check_tangent_vector(state.M, state.x, grad_fgₖ)

    ## 2. Get Truncated Newton direction
    ϵ_residual = min(0.5, sqrt(norm_rgrad)) * norm_rgrad    # Forcing sequence as sugested in NW, p. 168
    dᴺ, man_truncnewton_state.CG_niter = solve_tCG(state.M, state.x, grad_fgₖ, hessfg_x_h, ϵ_residual=ϵ_residual, ν = νₖ)

    check_tangent_vector(state.M, state.x, dᴺ)

    ## 3. Execute linesearch
    hist_ls = Dict()

    x_ls = linesearch(o.linesearch, pb, state.M, state.x, grad_fgₖ, dᴺ, hist=hist_ls)

    ncalls_f += hist_ls[:ncalls_f]

    ## 4. Update CG precision
    ν_strat = :ls
    if ν_strat == :ls
        if hist_ls[:niter] == 1
            man_truncnewton_state.νₖ *= ν_reductionfactor
        end
    elseif ν_strat == :lsnormgrad
        if hist_ls[:niter] == 1 && νₖ ≥ 1e-2 * norm_rgrad^2
            man_truncnewton_state.νₖ *= ν_reductionfactor
        end
    else
        @error "unknown reduction strategy"
    end

    state.x .= x_ls

    ### Logging data
    man_truncnewton_state.norm_∇fgₘ = norm(state.M, state.x, grad_fgₖ)
    man_truncnewton_state.CG_ε = ϵ_residual
    man_truncnewton_state.CG_residual = norm(state.M, state.x, hessfg_x_h(dᴺ) + grad_fgₖ)
    man_truncnewton_state.norm_dᴺ = norm(state.M, state.x, dᴺ)
    man_truncnewton_state.cosθ = inner(state.M, state.x, -dᴺ, grad_fgₖ)/(norm(state.M, state.x, dᴺ)*norm_rgrad)
    man_truncnewton_state.λ_x = inner(state.M, state.x, -dᴺ, grad_fgₖ)
    man_truncnewton_state.ls_niter = hist_ls[:niter]

    return
end
