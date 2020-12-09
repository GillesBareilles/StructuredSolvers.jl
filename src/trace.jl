@with_kw struct OptimizationState{T} <: OptimizerTrace
    it::Int64
    time::Float64 = 0.0
    f_x::Float64
    g_x::Float64
    norm_step::Float64 = -1.0
    norm_minsubgradient_tangent::Float64 = -1.0
    norm_minsubgradient_normal::Float64 = -1.0
    ncalls_f::Int64 = -1
    ncalls_g::Int64 = -1
    ncalls_∇f::Int64 = -1
    ncalls_proxg::Int64 = -1
    ncalls_gradₘF::Int64 = -1
    ncalls_HessₘF::Int64 = -1
    ncalls_retr::Int64 = -1
    niter_CG::Int64 = -1
    niter_manls::Int64 = -1
    additionalinfo::T = NamedTuple()
end

const OptimizationTrace{T} = Vector{OptimizationState{T}}
