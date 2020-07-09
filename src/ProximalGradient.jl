
struct ProximalGradient{T} <: Optimizer
    backtracking::Bool
    extrapolation::T
end
function ProximalGradient(; backtracking = true, extrapolation = VanillaProxGrad())
    return ProximalGradient(backtracking, extrapolation)
end

function Base.show(io::IO, o::ProximalGradient)
    print(io, "Proximal Gradient ")
    o.backtracking && print(io, "(bt)")
    show(io, o.extrapolation)
    return
end



mutable struct ProximalGradientState{Tx,Te} <: OptimizerState
    it::Int64
    x::Tx
    x_old::Tx
    M::Manifold
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
) where {Tx,O,Te,R}
    return ProximalGradientState{Tx,Te}(
        0,
        copy(x),
        copy(x),
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
    state.∇f_x = ∇f(pb, state.x)
    state.it += 1
    return
end

initial_state(o::ProximalGradient, x, reg) = ProximalGradientState(o, x, reg)

function update_iterate!(state::ProximalGradientState, pb, optimizer::ProximalGradient)

    ## Run backtracking line search to update estimate of gradient Lipschitz constant
    if optimizer.backtracking
        backtrack_f_lipschitzgradient!(state, pb)
    end

    state.x_old .= state.x
    extrapolation!(optimizer.extrapolation, state, state.extrapolation_state, pb)
    return
end


function display_logs_header(o::ProximalGradient)
    println("it.   F(x)                    f(x)       g(x)       Manifold      γ   step")
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
    tracing &&
        @printf "%4i  %.16e  %-.3e  %-.3e  %-12s  %-.3e  %.3e\n" iteration state.f_x +
                                                                           state.g_x state.f_x state.g_x state.M state.γ norm(
            state.x - state.x_old,
        )


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






function backtrack_f_lipschitzgradient!(state, pb)
    τ = 1.2

    ∇f_norm2 = norm(state.∇f_x)^2

    itmax = 200
    ncalls_f = 0

    it_ls = 0
    while it_ls <= itmax
        state.temp .= state.x .- state.γ .* state.∇f_x

        # f(pb, state.temp) ≤ state.f_x - 1/(2*state.γ) * norm(state.temp-state.x)^2 && break
        f(pb, state.temp) ≤ state.f_x - state.γ / 2 * ∇f_norm2 && break

        ncalls_f += 1
        state.γ = state.γ / τ
        it_ls += 1
    end

    if it_ls > itmax
        @warn "Gradient backtracking: reached iterations limits."
    end


    return ncalls_f
end
