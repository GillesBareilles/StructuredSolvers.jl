###########################################################
### Manifold Truncated Newton
###########################################################

abstract type TruncationStrategy end
@with_kw struct Newton <: TruncationStrategy
    ε_CGres::Float64 = 1e-13
    ν_CGreductionfactor::Float64 = 1e-10
end
update_εν(n::Newton, norm_rgrad) = n.ε_CGres, n.ν_CGreductionfactor


struct TruncatedNewton <: TruncationStrategy
end
function update_εν(::TruncatedNewton, norm_rgrad)
    return (
        min(0.5, sqrt(norm_rgrad)) * norm_rgrad,    # Forcing sequence as sugested in NW, p. 168
        1e-10
    )
end


"""
    ManTruncatedNewton

Optimizer performing inexact Newton steps on a prescribed manifold. The inexactness in
the approximation of the Newton step is controlled by `νₖ`: one expects a direction `dᴺ`
such that
        ||∇²(f+g)(x)[dᴺ] + ∇(f+g)|| ≤ νₖ ||∇(f+g)||
"""
@with_kw struct ManTruncatedNewton{T} <: ManifoldUpdate
    linesearch::ManifoldLinesearch = ArmijoGoldstein()
    ν_reductionfactor::Float64 = 0.5
    CG_maxiter::Int64 = 400
    truncationstrat::T = TruncatedNewton()
end
Base.summary(::ManTruncatedNewton{Newton}) = "ManNewton"
Base.summary(::ManTruncatedNewton{TruncatedNewton}) = "ManTruncatedNewton"


ManNewton(kwargs...) = ManTruncatedNewton(truncationstrat=Newton(), kwargs...)



@with_kw mutable struct ManTruncatedNewtonState <: AbstractUpdateState
    norm_∇fgₘ::Float64 = -1.0
    νₖ::Float64 = 1.0
    CG_niter::Int64 = -1
    CG_ε::Float64 = -1
    CG_residual::Float64 = -1
    d_type::Symbol = :Unsolved
    norm_dᴺ::Float64 = -1
    cosθ::Float64 = -1
    λ_x::Float64 = -1
    ls_niter::Int64 = -1
end

initial_state(::ManTruncatedNewton, x, reg) = ManTruncatedNewtonState()

# str_updatelog(o::ManifoldGradient, t::ManifoldGradientState) = @sprintf "ls-nit %2i\t\t||gradₘ f+g|| %.3e" t.ls_niter t.norm_∇fgₘ

function str_updatelog(::ManTruncatedNewton, t::ManTruncatedNewtonState)
    resstyle = (t.CG_residual <= t.CG_ε) ? "0" : "4;1"
    return string(
        (@sprintf "CG: nit:%3i  %s νₖ:%.1e ε:%.1e res:" t.CG_niter string(t.d_type) t.νₖ t.CG_ε),
        "\033[$(resstyle)m",
        (@sprintf "%.1e" t.CG_residual),
        "\033[0m",
        (@sprintf "     |dᴺ|:%.3e   cos(θₖ):%.1e  λ(x):%.1e   ls-nit%2i" t.norm_dᴺ t.cosθ t.λ_x t.ls_niter)
    )
end

function update_iterate!(state::PartlySmoothOptimizerState{Tx}, pb, o::ManTruncatedNewton) where Tx
    ###
    grad_fgₖ = state.tempvec_man
    M = state.M
    x = state.x.man_repr

    @assert state.x.repr == manifold_repr
    @assert is_manifold_point(M, x)

    state_TN = state.update_to_updatestate[o]

    @unpack ν_reductionfactor = o
    @unpack νₖ = state_TN

    ## 1. Get first & second order information
    # TODO: remove intermediate alloc from .+= op.
    grad_fgₖ = egrad_to_rgrad(M, x, state.∇f_x) + ∇M_g(pb, M, x); state.ncalls_gradₘF += 1
    function hessfg_x_h(ξ)
        state.ncalls_HessₘF += 1
        return ehess_to_rhess(M, x, state.∇f_x, ∇²f_h(pb, x, ξ), ξ) + ∇²M_g_ξ(pb, M, x, ξ)
    end

    norm_rgrad = norm(M, x, grad_fgₖ);
    state_TN.norm_∇fgₘ = norm_rgrad
    # check_tangent_vector(M, x, grad_fgₖ)


    ## 2. Get Truncated Newton direction
    ϵ_residual, νₖ = update_εν(o.truncationstrat, norm_rgrad)

    maxiter = 400
    dᴺ, state_TN.CG_niter, state_TN.d_type = solve_tCG(M, x, grad_fgₖ, hessfg_x_h, ϵ_residual = ϵ_residual, ν = νₖ, printlev=0, maxiter=maxiter)

    # check_tangent_vector(M, x, dᴺ)

    ## 3. Execute linesearch
    # TODO: make linesearch inplace for x, return status.
    hist_ls = Dict()
    x_ls = linesearch(o.linesearch, state, pb, M, x, grad_fgₖ, dᴺ, hist=hist_ls)

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
    state_TN.CG_ε = ϵ_residual
    state_TN.CG_residual = norm(M, x, hessfg_x_h(dᴺ) + grad_fgₖ)
    state_TN.norm_dᴺ = norm(M, x, dᴺ)
    state_TN.cosθ = inner(M, x, -dᴺ, grad_fgₖ)/(norm(M, x, dᴺ)*norm_rgrad)
    state_TN.λ_x = inner(M, x, -dᴺ, grad_fgₖ)
    state_TN.ls_niter = hist_ls[:niter]

    return
end
