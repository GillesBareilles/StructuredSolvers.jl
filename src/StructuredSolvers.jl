module StructuredSolvers

# Write your package code here.

using Manifolds
using StructuredProximalOperators
using CompositeProblems
using DataStructures
using Printf
using TimerOutputs
using Parameters

import Base: show, summary, copy
import CompositeProblems: firstorder_optimality_tangnorm
import StructuredProximalOperators: embed

const to = TimerOutput()

abstract type Optimizer end
abstract type ManifoldOptimizer <: Optimizer end
abstract type AmbiantOptimizer <: Optimizer end

abstract type OptimizerState end
abstract type OptimizerTrace end

abstract type ConvergenceChecker end


export OptimizerParams

include("trace.jl")
include("optimizer_params.jl")
include("optimize.jl")

include("point_ambiantmanifold.jl")


#
### Proximal gradient based algorithms
#
include("ProximalGradient.jl")
include("ProximalGradient_extrapolations.jl")


#
### ProxGrad / Manifold based algorithms
#
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
abstract type ManifoldLinesearch end
include("linesearches/armijogoldstein.jl")
include("manifold_updates/tangent_CG.jl")

include("manifold_updates/manifold_identity.jl")
include("manifold_updates/manifold_gradient.jl")
include("manifold_updates/manifold_truncatednewton.jl")
include("manifold_updates/manifold_NewtonCG.jl")
export ManifoldIdentity
export ManifoldGradient
export ManifoldTruncatedNewton
export ManNewtonCG


using Colors
using PGFPlotsX
using LaTeXStrings
using Contour

include("plots/plot_trace.jl")
include("plots/plot_highlevel.jl")

include("utils_numexps.jl")

include("convergence.jl")

function firstorder_optimality_tangnorm(pb::CompositeProblem, x::Point, M, ∇f_x)
    return firstorder_optimality_tangnorm(pb, get_repr(x), M, ∇f_x)
end


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
