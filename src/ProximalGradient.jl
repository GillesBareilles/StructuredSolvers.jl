
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
    return
end

initial_state(o::ProximalGradient, x, reg) = ProximalGradientState(o, x, reg)

function update_iterate!(state::ProximalGradientState, pb, optimizer::ProximalGradient)
    state.M_old = state.M

    ## Run backtracking line search to update estimate of gradient Lipschitz constant
    if optimizer.backtracking
        ncalls_f, state.γ = backtrack_f_lipschitzgradient!(state, pb, state.γ)
    end

    state.x_old .= state.x
    extrapolation!(optimizer.extrapolation, state, state.extrapolation_state, pb)
    return
end


function display_logs_header(o::ProximalGradient)
    println("it.   F(x)                    f(x)       g(x)       γ          step       Manifold")
    return
end

function display_logs(
    state::ProximalGradientState,
    pb,
    optimizer,
    iteration,
    time,
    optimstate_extensions,
    tracing,
)
    if tracing
        style = state.M==state.M_old ? "" : "4"
        @printf "%4i  %.16e  %-.3e  %-.3e  %.3e  %-.3e" iteration state.f_x + state.g_x state.f_x state.g_x state.γ norm(state.x - state.x_old)

        if state.M == state.M_old
            @printf "  \033[m%s\033[0m\n" state.M
        else
            @printf "  \033[4m%s\033[0m\n" state.M
        end
    end

    ## TODO: This section of code is generic, should be factored. Plus, can the many allocs be avoided when zipping and building temp arrays?
    keys = []
    values = []
    for osextension in optimstate_extensions
        push!(keys)
        push!(values, osextension.getvalue(state))
    end

    return OptimizationState(
        it = iteration,
        time = time,
        f_x = state.f_x,
        g_x = state.g_x,
        additionalinfo = (;
            zip(
                [osextension.key for osextension in optimstate_extensions],
                [
                    copy(osextension.getvalue(state))
                    for osextension in optimstate_extensions
                ],
            )...),
    )
end






function backtrack_f_lipschitzgradient!(state, pb, γ)
    τ = 1.2

    ∇f_norm2 = norm(state.∇f_x)^2

    itmax = 200
    ncalls_f = 0

    it_ls = 0
    while it_ls <= itmax
        state.temp .= state.x .- γ .* state.∇f_x

        # f(pb, state.temp) ≤ state.f_x - 1/(2*γ) * norm(state.temp-state.x)^2 && break
        f(pb, state.temp) ≤ state.f_x - γ / 2 * ∇f_norm2 && break

        ncalls_f += 1
        γ = γ / τ
        it_ls += 1
    end

    if it_ls > itmax
        @warn "Gradient backtracking: reached iterations limits."
    end


    return ncalls_f, γ
end
