
#
## Vanilla proximal gradient
#
abstract type ProxGradExtrapolation end
abstract type ProxGradExtrapolationState <: OptimizerState end


struct VanillaProxGrad <: ProxGradExtrapolation end
mutable struct VanillaProxGradState <: ProxGradExtrapolationState end

extrapolation_state(::VanillaProxGrad, x, g) = VanillaProxGradState()
function extrapolation!(::VanillaProxGrad, pgstate, extrapolationstate, pb)
    pgstate.temp .= pgstate.x .- pgstate.γ .* pgstate.∇f_x
    M = prox_αg!(pb, pgstate.x, pgstate.temp, pgstate.γ)
    pgstate.M = M
    return
end

function Base.show(io::IO, ::VanillaProxGrad)
    return print(io, "")
end

#
## Accelerated proximal gradient
#
## Inertial parameters from FISTA, but with possibilities suggested by Jingwei Liang...
@with_kw struct AcceleratedProxGrad <: ProxGradExtrapolation
    p::Float64 = 1.0
    q::Float64 = 1.0
    r::Float64 = 4.0
end
mutable struct AcceleratedProxGradState{Tx} <: ProxGradExtrapolation
    t::Float64
    y::Tx
    y_old::Tx
end

function Base.show(io::IO, ::AcceleratedProxGrad)
    return print(io, " Accelerated")
end

function extrapolation_state(::AcceleratedProxGrad, x, g)
    return AcceleratedProxGradState(1.0, zero(x), zero(x))
end
function extrapolation!(apg::AcceleratedProxGrad, pgstate, extrastate, pb)
    @unpack p, q, r = apg

    # y = proxgrad(x)
    pgstate.temp .= pgstate.x .- pgstate.γ .* pgstate.∇f_x
    M = prox_αg!(pb, extrastate.y, pgstate.temp, pgstate.γ)
    pgstate.M = M

    t_next = (p + sqrt(q + r * extrastate.t^2)) / 2

    # x = y + α (y-y_old)
    pgstate.x .=
        extrastate.y .+ (extrastate.t - 1) / t_next * (extrastate.y .- extrastate.y_old)
    extrastate.t = t_next
    extrastate.y_old .= extrastate.y

    return
end


#
## T1
#
@with_kw struct Test1ProxGrad <: ProxGradExtrapolation
    acceleration::AcceleratedProxGrad = AcceleratedProxGrad()
end
mutable struct Test1ProxGradState <: ProxGradExtrapolationState
    M_previousit::Manifold
    acceleration_state::AcceleratedProxGradState
end

function extrapolation_state(o::Test1ProxGrad, x, g::Regularizer)
    return Test1ProxGradState(
        wholespace_manifold(g, x),
        extrapolation_state(o.acceleration, x, g),
    )
end
function extrapolation!(o::Test1ProxGrad, pgstate, extrastate, pb)
    @unpack p, q, r = o.acceleration
    accel_state = extrastate.acceleration_state

    ## Compute proximal gradient iterate
    pgstate.temp .= pgstate.x .- pgstate.γ .* pgstate.∇f_x
    M = prox_αg!(pb, accel_state.y, pgstate.temp, pgstate.γ)

    t_next = (p + sqrt(q + r * accel_state.t^2)) / 2

    ## Compute the possibly accelerated iterate
    if !(M < extrastate.M_previousit)
        pgstate.x .=
            accel_state.y .+
            (accel_state.t - 1) / t_next * (accel_state.y .- accel_state.y_old)
    else
        pgstate.x .= accel_state.y
    end

    pgstate.M = M
    extrastate.M_previousit = M

    accel_state.t = t_next
    accel_state.y_old .= accel_state.y

    return
end

function Base.show(io::IO, ::Test1ProxGrad)
    return print(io, " - Test 1")
end

#
## T2
#
@with_kw struct Test2ProxGrad <: ProxGradExtrapolation
    acceleration::AcceleratedProxGrad = AcceleratedProxGrad()
end
mutable struct Test2ProxGradState{Tx} <: ProxGradExtrapolationState
    M_previousit::Manifold
    acceleration_state::AcceleratedProxGradState{Tx}
    candidate_pg::Tx
    candidate_apg::Tx
end

function extrapolation_state(o::Test2ProxGrad, x, g::Regularizer)
    return Test2ProxGradState(
        wholespace_manifold(g, x),
        extrapolation_state(o.acceleration, x, g),
        zero(x),
        zero(x),
    )
end
function extrapolation!(o::Test2ProxGrad, pgstate, extrastate, pb)
    @unpack p, q, r = o.acceleration
    accel_state = extrastate.acceleration_state


    ## Compute proximal gradient iterate
    pgstate.temp .= pgstate.x .- pgstate.γ .* pgstate.∇f_x
    M_accelerated = prox_αg!(pb, accel_state.y, pgstate.temp, pgstate.γ)

    t_next = (p + sqrt(q + r * accel_state.t^2)) / 2

    extrastate.candidate_pg .= accel_state.y
    extrastate.candidate_apg .=
        accel_state.y .+ (accel_state.t - 1) / t_next * (accel_state.y .- accel_state.y_old)

    # Compute image of candidates by PG, only manifold is relevant.
    ## 1. proxgrad
    pgstate.temp .= extrastate.candidate_pg .- pgstate.γ .* ∇f(pb, extrastate.candidate_pg)
    M_PG_pg = prox_αg!(pb, pgstate.temp, pgstate.temp, pgstate.γ)

    pgstate.temp .=
        extrastate.candidate_apg .- pgstate.γ .* ∇f(pb, extrastate.candidate_apg)
    M_PG_apg = prox_αg!(pb, pgstate.temp, pgstate.temp, pgstate.γ)

    ## Here, if apg has less structure than pg (in image by proxgrad), no acceleration
    if !(M_PG_pg < M_PG_apg)
        pgstate.x .= extrastate.candidate_apg
        pgstate.M = M_PG_pg
    else
        pgstate.x .= extrastate.candidate_pg
        pgstate.M = M_PG_apg
    end

    extrastate.M_previousit = pgstate.M

    accel_state.t = t_next
    accel_state.y_old .= accel_state.y

    return
end

function Base.show(io::IO, ::Test2ProxGrad)
    return print(io, " - Test 2")
end




#
## MFISTA
#
@with_kw struct MFISTA <: ProxGradExtrapolation
    acceleration::AcceleratedProxGrad = AcceleratedProxGrad()
end
mutable struct MFISTAState{Tx} <: ProxGradExtrapolationState
    acceleration_state::AcceleratedProxGradState{Tx}
    z::Tx
    z_old::Tx
end

function extrapolation_state(o::MFISTA, x, g::Regularizer)
    return MFISTAState(extrapolation_state(o.acceleration, x, g), copy(x), copy(x))
end

"""
    extrapolation!(o::MFISTA, pgstate, extrastate, pb)

MFISTA algorithm.

NOTE: implementaiton might be improved.
"""
function extrapolation!(o::MFISTA, pgstate, extrastate, pb)
    @unpack p, q, r = o.acceleration
    accel_state = extrastate.acceleration_state

    ## Compute proximal gradient iterate
    pgstate.temp .= pgstate.x .- pgstate.γ .* pgstate.∇f_x
    M = prox_αg!(pb, accel_state.y, pgstate.temp, pgstate.γ)

    t_next = (p + sqrt(q + r * accel_state.t^2)) / 2


    if F(pb, accel_state.y) <= F(pb, extrastate.z_old)
        extrastate.z .= accel_state.y
    else
        extrastate.z .= extrastate.z_old
    end

    pgstate.x .=
        extrastate.z .+ accel_state.t / t_next .* (accel_state.y .- extrastate.z) .+
        (accel_state.t - 1) / t_next .* (extrastate.z .- extrastate.z_old)
    pgstate.M = M

    extrastate.z_old .= extrastate.z
    accel_state.t = t_next

    return
end

function Base.show(io::IO, ::MFISTA)
    return print(io, "MFISTA")
end
