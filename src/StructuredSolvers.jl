module StructuredSolvers

# Write your package code here.

using StructuredProximalOperators
using CompositeProblems
using DataStructures
using Printf
using TimerOutputs
using Parameters

const to = TimerOutput()

abstract type Optimizer end
abstract type ManifoldOptimizer <: Optimizer end
abstract type AmbiantOptimizer <: Optimizer end

abstract type OptimizerState end
abstract type OptimizerTrace end


include("trace.jl")
include("optimize.jl")
include("ProximalGradient.jl")

using PGFPlotsX, LaTeXStrings

include("plot_trace.jl")


export ProximalGradient, ProximalGradientState
export VanillaProxGrad, VanillaProxGradState
export AcceleratedProxGrad, AcceleratedProxGradState

export OptimizationState, OptimizationTrace

export optimize!

end
