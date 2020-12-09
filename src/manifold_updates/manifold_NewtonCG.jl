###########################################################
### Manifold Truncated Newton
###########################################################
"""
    ManNewtonCG

Optimizer performing inexact Newton steps on a prescribed manifold. The inexactness in
the approximation of the Newton step is controlled by `νₖ`: one expects a direction `dᴺ`
such that
        ||∇²(f+g)(x)[dᴺ] + ∇(f+g)|| ≤ νₖ ||∇(f+g)||
"""
@with_kw struct ManNewtonCG <: ManifoldUpdate
    linesearch::ManifoldLinesearch = ArmijoGoldstein()
    # ν_reductionfactor::Float64 = 0.5
end
Base.summary(::ManNewtonCG) = "ManNewtonCG"

@with_kw mutable struct ManNewtonCGState <: AbstractUpdateState
    norm_∇fgₘ::Float64 = -1.0
    d_type::Symbol = :Unsolved
    νₖ::Float64 = 1.0
    CG_niter::Int64 = -1
    CG_ε::Float64 = -1
    CG_residual::Float64 = -1
    norm_dᴺ::Float64 = -1
    cosθ::Float64 = -1
    λ_x::Float64 = -1
    ls_niter::Int64 = -1
end

initial_state(::ManNewtonCG, x, reg) = ManNewtonCGState()

# str_updatelog(o::ManifoldGradient, t::ManifoldGradientState) = @sprintf "ls-nit %2i\t\t||gradₘ f+g|| %.3e" t.ls_niter t.norm_∇fgₘ

function str_updatelog(::ManNewtonCG, t::ManNewtonCGState)
    resstyle = (t.CG_residual <= t.CG_ε) ? "0" : "4;1"
    return string(
        (@sprintf "%-17s CG: nit:%3i  ε:%.1e res:" t.d_type t.CG_niter t.CG_ε),
        "\033[$(resstyle)m",
        (@sprintf "%.1e" t.CG_residual),
        "\033[0m",
        (@sprintf "     |dᴺ|:%.3e   cos(θₖ):%.1e  λ(x):%.1e  ls-nit %2i" t.norm_dᴺ t.cosθ t.λ_x t.ls_niter)
    )
end

function update_iterate!(state::PartlySmoothOptimizerState{Tx}, pb, o::ManNewtonCG) where Tx
    ###
    grad_fgₖ = state.tempvec_man
    M = state.M
    x = state.x.man_repr

    @assert state.x.repr == manifold_repr
    @assert is_manifold_point(M, x)

    state_TN = state.update_to_updatestate[o]

    # @unpack ν_reductionfactor = o
    ν_reductionfactor = 1.0
    @unpack νₖ = state_TN

    ## 1. Get first & second order information
    # TODO: remove intermediate alloc from .+= op.
    grad_fgₖ = egrad_to_rgrad(M, x, state.∇f_x) + ∇M_g(pb, M, x); state.ncalls_gradₘF += 1
    function hessfg_x_h(ξ)
        state.ncalls_HessₘF += 1
        return ehess_to_rhess(M, x, state.∇f_x, ∇²f_h(pb, x, ξ), ξ) + ∇²M_g_ξ(pb, M, x, ξ)
    end

    norm_rgrad = norm(M, x, grad_fgₖ)
    state_TN.norm_∇fgₘ = norm_rgrad

    # check_tangent_vector(M, x, grad_fgₖ)

    ## 2. Get Truncated Newton direction
    # TODO: turn this into approximalte solution finding
    dₖ, state_TN.d_type, state_TN.CG_niter = solve_tCG_capped(M, x, grad_fgₖ, hessfg_x_h; ϵ=1e-15, ζ = 1e-15, maxiter=1e5, check_tvectors=false, printlev=0)

    if state_TN.d_type == :NegativeCurvature
        dₖ = - sign(inner(M, x, dₖ, grad_fgₖ)) * abs(inner(M, x, dₖ, hessfg_x_h(dₖ))) * norm(M, x, dₖ)^(-3) * dₖ
    elseif state_TN.d_type == :MaxIter
    else
        @assert state_TN.d_type == :Solution "Unknown d_type" state_TN.d_type
    end


    # check_tangent_vector(M, x, dᴺ)

    ## 3. Execute linesearch
    # TODO: make linesearch inplace for x, return status.
    hist_ls = Dict()
    x_ls = linesearch(o.linesearch, state, pb, M, x, grad_fgₖ, dₖ, hist=hist_ls)

    x = x_ls
    state_TN.ls_niter = hist_ls[:niter]



    ## 4. Update CG precision
    ν_strat = :ls
    if ν_strat == :ls
        if hist_ls[:niter] == 1
            state_TN.νₖ *= ν_reductionfactor
        end
    elseif ν_strat == :lsnormgrad
        if hist_ls[:niter] == 1 && νₖ ≥ 1e-2 * norm_rgrad^2
            state_TN.νₖ *= ν_reductionfactor
        end
    else
        @error "unknown reduction strategy"
    end

    state.x.man_repr = x_ls

    ### Logging data
    # state_TN.CG_ε = ϵ_residual
    state_TN.CG_residual = norm(M, x, hessfg_x_h(dₖ) + grad_fgₖ)
    state_TN.norm_dᴺ = norm(M, x, dₖ)
    state_TN.cosθ = inner(M, x, -dₖ, grad_fgₖ)/(norm(M, x, dₖ)*norm_rgrad)
    state_TN.λ_x = inner(M, x, -dₖ, grad_fgₖ)
    state_TN.ls_niter = hist_ls[:niter]

    return
end
