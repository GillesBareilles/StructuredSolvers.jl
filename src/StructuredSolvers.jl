module StructuredSolvers

# Write your package code here.

using Manifolds
using StructuredProximalOperators
using CompositeProblems
using DataStructures
using Printf
using TimerOutputs
using Parameters

import Base: show, summary

const to = TimerOutput()

abstract type Optimizer end
abstract type ManifoldOptimizer <: Optimizer end
abstract type AmbiantOptimizer <: Optimizer end

abstract type OptimizerState end
abstract type OptimizerTrace end


include("trace.jl")
include("optimize.jl")
include("ProximalGradient.jl")
include("ProximalGradient_extrapolations.jl")

include("PartlySmoothOptimizer.jl")
export PartlySmoothOptimizer

# Update strategies
include("update_selectors/alternatingUpdateSelector.jl")
include("update_selectors/manifold_following.jl")
include("update_selectors/constant_manifold.jl")
export AlternatingUpdateSelector
export ManifoldFollowingSelector
export ConstantManifoldSelector

# Wholespace updates
include("wholespace_updates/proximal_gradient.jl")
export WholespaceProximalGradient

# Manifold updates
include("linesearches.jl")
include("manifold_updates/tangent_ConjugatedGradient.jl")

include("manifold_updates/manifold_identity.jl")
include("manifold_updates/manifold_gradient.jl")
include("manifold_updates/manifold_truncatednewton.jl")
export ManifoldIdentity
export ManifoldGradient
export ManifoldTruncatedNewton


using Colors
using PGFPlotsX
using LaTeXStrings
using Contour

include("plot_trace.jl")


export ProximalGradient, ProximalGradientState
export VanillaProxGrad, VanillaProxGradState
export AcceleratedProxGrad, AcceleratedProxGradState
export Test1ProxGrad, Test1ProxGradState
export Test2ProxGrad, Test2ProxGradState
export MFISTA, MFISTAState
export RestartedAPG, RestartedAPGState

export OptimizationState, OptimizationTrace
export Optimizer

export optimize!

end
