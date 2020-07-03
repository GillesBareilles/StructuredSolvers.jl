
struct ProximalGradient{T} <: Optimizer
    backtracking::Bool
    extrapolation::T
end
function ProximalGradient(; backtracking = true, extrapolation = VanillaProxGrad())
    return ProximalGradient(backtracking, extrapolation)
end

mutable struct ProximalGradientState{Tx,Te} <: OptimizerState
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
    x::Tx;
    γ = 1e5,
    extra_state::Te = extrapolation_state(o.extrapolation, x),
) where {Tx,O,Te}
    return ProximalGradientState{Tx,Te}(
        zero(x),
        zero(x),
        Euclidean(size(x)...),
        0.0,
        0.0,
        zero(x),
        zero(x),
        γ,
        extra_state,
    )
end

# struct GenericTrace <: OptimizerTrace
#     it::Int64
#     time::Float64
#     f_x::Float64
#     g_x::Float64
#     norm_∇f_x::Float64
#     nb_calls_f::Int64
#     nb_calls_g::Int64
#     nb_calls_∇f::Int64
#     nb_calls_proxg::Int64
#     nb_calls_∇²fh::Int64
#     nb_calls_∇²gξ::Int64
# end

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
    return
end

initial_state(o::ProximalGradient, x) = ProximalGradientState(o, x)

function update_iterate!(state::ProximalGradientState, pb, optimizer::ProximalGradient)

    ## Run backtracking line search to update estimate of gradient Lipschitz constant
    if optimizer.backtracking
        backtrack_f_lipschitzgradient!(state, pb)
    end

    state.x_old .= state.x
    extrapolation!(optimizer.extrapolation, state, pb)
    return
end


function display_logs_header(o::ProximalGradient)
    println("it.   F(x)                    f(x)       g(x)       Manifold      γ   step")
    return
end

function display_logs(state::ProximalGradientState, pb, optimizer, iteration)
    @printf "%4i  %.16e  %-.3e  %-.3e  %-12s  %-.3e  %.3e\n" iteration state.f_x +
                                                                              state.g_x state.f_x state.g_x state.M state.γ norm(
        state.x - state.x_old,
    )
    return
end



######
abstract type ProxGradExtrapolation end
abstract type ProxGradExtrapolationState <: OptimizerState end


struct VanillaProxGrad <: ProxGradExtrapolation end
mutable struct VanillaProxGradState <: ProxGradExtrapolationState end

extrapolation_state(::VanillaProxGrad, x) = VanillaProxGradState()
function extrapolation!(::VanillaProxGrad, state, pb)
    state.temp .= state.x .- state.γ .* state.∇f_x
    M = prox_αg!(pb, state.x, state.temp, state.γ)
    state.M = M
    return
end


struct AcceleratedProxGrad <: ProxGradExtrapolation end
mutable struct AcceleratedProxGradState{Tx} <: ProxGradExtrapolation
    t::Float64
    y::Tx
    y_old::Tx
end

function extrapolation_state(::AcceleratedProxGrad, x)
    return AcceleratedProxGradState(1.0, zero(x), zero(x))
end
function extrapolation!(::AcceleratedProxGrad, state, pb)
    extra_state = state.extrapolation_state

    state.temp .= state.x .- state.γ .* state.∇f_x
    M = prox_αg!(pb, extra_state.y, state.temp, state.γ)
    state.M = M

    t_next = 1 + sqrt(1 + 4 * extra_state.t^2)

    state.x .=
        extra_state.y + (extra_state.t - 1) / t_next * (extra_state.y - extra_state.y_old)
    extra_state.t = t_next
    extra_state.y_old .= extra_state.y

    return
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
