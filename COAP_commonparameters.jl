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
