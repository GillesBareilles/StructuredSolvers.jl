
struct ProximalGradient{T} <: Optimizer
    backtracking::Bool
    extrapolation::T
end
function ProximalGradient(; backtracking = true, extrapolation = VanillaProxGrad())
    return ProximalGradient(backtracking, extrapolation)
end

Base.summary(o::ProximalGradient) = string(
    "Proximal Gradient ",
    o.backtracking && "(bt)",
    summary(o.extrapolation)
)


mutable struct ProximalGradientState{Tx,Te} <: OptimizerState
    it::Int64
    x::Tx
    x_old::Tx
    M::Manifold
    M_old::Manifold
    f_x::Float64
    g_x::Float64
    ∇f_x::Tx
    temp::Tx
    γ::Float64
    extrapolation_state::Te
    ncalls_f::Int64
    ncalls_g::Int64
    ncalls_∇f::Int64
    ncalls_proxg::Int64
end
function ProximalGradientState(
    o::ProximalGradient,
    x::Tx,
    g::R;
    γ = 1e5,
    extra_state::Te = extrapolation_state(o.extrapolation, x, g),
) where {Tx,Te,R}
    return ProximalGradientState{Tx,Te}(
        0,
        copy(x),
        copy(x),
        wholespace_manifold(g, x),
        wholespace_manifold(g, x),
        0.0,
        0.0,
        zero(x),
        zero(x),
        γ,
        extra_state,
        0, 0, 0, 0
    )
end


function print_header(pg::ProximalGradient)
    println("---------------------")
    println("--- Proximal Gradient")
    println(" - backtracking:       ", pg.backtracking)
    println(" - extrapolation:       ", pg.extrapolation)
    return
end

function update_fg∇f!(state::ProximalGradientState, pb)
    state.f_x = f(pb, state.x)
    state.g_x = g(pb, state.x)
    ∇f!(pb, state.∇f_x, state.x)
    state.it += 1
    state.ncalls_∇f += 1
    state.ncalls_f += 1
    state.ncalls_g += 1
    return
end

initial_state(o::ProximalGradient, x, reg; manifold=nothing) = ProximalGradientState(o, x, reg)

function update_iterate!(state::ProximalGradientState, pb, optimizer::ProximalGradient)

    ## Run backtracking line search to update estimate of gradient Lipschitz constant
    if optimizer.backtracking
        state.γ = backtrack_f_lipschitzgradient!(state, pb, state.γ)
    end

    extrapolation!(optimizer.extrapolation, state, state.extrapolation_state, pb)
    return
end

function build_optimstate(::ProximalGradient, state, iteration, time, normstep, minsubgradient_tan, minsubgradient_norm, optimstate_extensions)
    return OptimizationState(
        it = iteration,
        time = time,
        f_x = state.f_x,
        g_x = state.g_x,
        norm_step = normstep,
        ncalls_f = state.ncalls_f,
        ncalls_g = state.ncalls_g,
        ncalls_∇f = state.ncalls_∇f,
        ncalls_proxg = state.ncalls_proxg,
        # ncalls_gradₘF = 0,
        # ncalls_HessₘF = 0,
        norm_minsubgradient_tangent = minsubgradient_tan,
        norm_minsubgradient_normal = minsubgradient_norm,
        additionalinfo = (;
            zip(
                [osextension.key for osextension in optimstate_extensions],
                [
                    copy(osextension.getvalue(state))
                    for osextension in optimstate_extensions
                ],
            )...
        ),
    )
end



display_logs_header_post(::ProximalGradient) = print("γ")

function display_logs_post(::ProximalGradient, state, pb)
    @printf "%.3e" state.γ
    return
end




function backtrack_f_lipschitzgradient!(state, pb, γ)
    τ = 1.2

    ∇f_norm2 = norm(state.∇f_x)^2

    itmax = 200

    it_ls = 0
    while it_ls <= itmax
        state.temp .= state.x .- γ .* state.∇f_x

        # f(pb, state.temp) ≤ state.f_x - 1/(2*γ) * norm(state.temp-state.x)^2 && break
        state.ncalls_f += 1
        f(pb, state.temp) ≤ state.f_x - γ / 2 * ∇f_norm2 && break

        γ = γ / τ
        it_ls += 1
    end

    if it_ls > itmax
        @warn "Gradient backtracking: reached iterations limits."
    end


    return γ
end
