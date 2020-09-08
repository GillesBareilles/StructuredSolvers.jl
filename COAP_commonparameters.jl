get_color(::ProximalGradient{VanillaProxGrad}) = "rgb,1:red,0.0;green,0.3843;blue,0.9922"
get_color(::ProximalGradient{AcceleratedProxGrad}) = "rgb,1:red,0.7451;green,0.0;blue,0.0"
get_color(::ProximalGradient{Test1ProxGrad}) = "rgb,1:red,0.0;green,0.502;blue,0.3804"
get_color(::ProximalGradient{Test2ProxGrad}) = "black"
get_color(::ProximalGradient{MFISTA}) = "rgb,255:red,231;green,84;blue,121"


StructuredSolvers.get_legendname(::ProximalGradient{VanillaProxGrad}) = "Proximal Gradient"
StructuredSolvers.get_legendname(::ProximalGradient{AcceleratedProxGrad}) = "Accel. Proximal Gradient"
StructuredSolvers.get_legendname(::ProximalGradient{Test1ProxGrad}) = "Prov. Alg -- \$\\mathsf{T}^1\$"
StructuredSolvers.get_legendname(::ProximalGradient{Test2ProxGrad}) = "Prov. Alg -- \$\\mathsf{T}^2\$"
StructuredSolvers.get_legendname(::ProximalGradient{MFISTA}) = "Monotone APG"


StructuredSolvers.get_curve_params(o::ProximalGradient{VanillaProxGrad}, COLORS, algoid, markrepeat) = Dict{Any, Any}(
    "mark" => "none",
    "color" => get_color(o),
    "dotted" => nothing
)
StructuredSolvers.get_curve_params(o::ProximalGradient{AcceleratedProxGrad}, COLORS, algoid, markrepeat) = Dict{Any, Any}(
    "mark" => "none",
    "color" => get_color(o),
    "densely dotted" => nothing
)
StructuredSolvers.get_curve_params(o::ProximalGradient{Test1ProxGrad}, COLORS, algoid, markrepeat) = Dict{Any, Any}(
    "mark" => "none",
    "color" => get_color(o),
    "dashed" => nothing
)
StructuredSolvers.get_curve_params(o::ProximalGradient{Test2ProxGrad}, COLORS, algoid, markrepeat) = Dict{Any, Any}(
    "mark" => "none",
    "color" => get_color(o),
)
StructuredSolvers.get_curve_params(o::ProximalGradient{MFISTA}, COLORS, algoid, markrepeat) = Dict{Any, Any}(
    "mark" => "none",
    "color" => get_color(o),
)



StructuredSolvers.get_iteratesplot_params(o::ProximalGradient{VanillaProxGrad}, COLORS, algoid) = Dict{Any, Any}(
    "smooth" => nothing,
    "very thick" => nothing,
    "mark size" => "1pt",
    "color" => get_color(o),
    "mark" => "*",
    "mark options" => "",
)
StructuredSolvers.get_iteratesplot_params(o::ProximalGradient{AcceleratedProxGrad}, COLORS, algoid) = Dict{Any, Any}(
    "smooth" => nothing,
    "very thick" => nothing,
    "mark size" => "1pt",
    "color" => get_color(o),
    "mark" => "triangle*",
    "mark options" => ""
)
StructuredSolvers.get_iteratesplot_params(o::ProximalGradient{Test1ProxGrad}, COLORS, algoid) = Dict{Any, Any}(
    "smooth" => nothing,
    "very thick" => nothing,
    # "fill opacity" => "0",
    "mark size" => "1.5pt",
    "color" => get_color(o),
    "mark" => "*",
    "mark options" => ""
)
StructuredSolvers.get_iteratesplot_params(o::ProximalGradient{Test2ProxGrad}, COLORS, algoid) = Dict{Any, Any}(
    "smooth" => nothing,
    "very thick" => nothing,
    # "fill opacity" => "0",
    "mark size" => "1.5pt",
    "color" => get_color(o),
    "mark" => "star",
    # "mark options" => ""
)
StructuredSolvers.get_iteratesplot_params(o::ProximalGradient{MFISTA}, COLORS, algoid) = Dict{Any, Any}(
    "smooth" => nothing,
    "very thick" => nothing,
    "mark size" => "1pt",
    "color" => get_color(o),
    "mark" => "pentagon*",
    "mark options" => ""
)





"""
    nb_identified_manifolds(M::Manifold)

Counts the number of "identified manifolds", monotone with manifold codimension. For l1,
number of zeros; for fixed rank, `min(m, n)-k`...
"""
nb_identified_manifolds(M::Manifold) = 0
nb_identified_manifolds(M::l1Manifold) = sum(1 .- M.nnz_coords)
nb_identified_manifolds(M::Euclidean) = 0
nb_identified_manifolds(M::PSphere) = 1
function nb_identified_manifolds(M::ProductManifold)
    return sum(nb_identified_manifolds(man) for man in M.manifolds)
end
nb_identified_manifolds(::FixedRankMatrices{m,n,k}) where {m,n,k} = min(m, n) - k

"""
nb_correctlyidentified_manifolds(M::Manifold, M_ref::Manifold)

Counts the number of "identified manifolds" of `M` present in `M_ref`.
"""
nb_correctlyidentified_manifolds(M::Manifold, M_ref::Manifold) = 0
function nb_correctlyidentified_manifolds(M::l1Manifold, M_ref::l1Manifold)
    return dot(1 .- M.nnz_coords, 1 .- M_ref.nnz_coords)
end
nb_correctlyidentified_manifolds(M::Euclidean, M_ref::Euclidean) = 0
nb_correctlyidentified_manifolds(M::PSphere, M_ref::PSphere) = M == M_ref
function nb_correctlyidentified_manifolds(M::ProductManifold, M_ref::ProductManifold)
    return sum(
        nb_correctlyidentified_manifolds(M.manifolds[i], M_ref.manifolds[i])
        for i in 1:length(M.manifolds)
    )
end
function nb_correctlyidentified_manifolds(
    ::FixedRankMatrices{m,n,k},
    ::FixedRankMatrices{m,n,k_ref},
) where {m,n,k,k_ref}
    return min(min(m, n) - k, min(m, n) - k_ref)
end


function get_proportion_identifiedstructure(Ms::AbstractVector, M_ref)
    return [nb_correctlyidentified_manifolds(M, M_ref) for M in Ms] ./ nb_identified_manifolds(M_ref)
end

function get_identification_ind(Ms, M_x0)
    ind_id = length(Ms)
    while Ms[ind_id] == M_x0
        ind_id -= 1
    end
    has_identified = ind_id != length(Ms)
    return has_identified, ind_id + 1
end

function get_finalmanifold_ind(Ms)
    ind_id = length(Ms)
    @show length(Ms)
    while Ms[ind_id] == Ms[end]
        ind_id -= 1
    end
    @show ind_id
    return ind_id + 1
end
