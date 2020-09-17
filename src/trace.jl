@with_kw struct OptimizationState{T} <: OptimizerTrace
    it::Int64
    time::Float64 = 0.0
    f_x::Float64
    g_x::Float64
    norm_∇f_x::Float64 = -1.0
    norm_step::Float64 = -1.0
    nb_calls_f::Int64 = -1
    nb_calls_g::Int64 = -1
    nb_calls_∇f::Int64 = -1
    nb_calls_proxg::Int64 = -1
    nb_calls_∇²fh::Int64 = -1
    nb_calls_∇²gξ::Int64 = -1
    additionalinfo::T = NamedTuple()
end

const OptimizationTrace{T} = Vector{OptimizationState{T}}
