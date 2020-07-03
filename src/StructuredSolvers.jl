module StructuredSolvers

# Write your package code here.

using StructuredProximalOperators
using CompositeProblems
using DataStructures
using Printf
using TimerOutputs

const to = TimerOutput()

abstract type Optimizer end
abstract type ManifoldOptimizer <: Optimizer end
abstract type AmbiantOptimizer <: Optimizer end

abstract type OptimizerState end
abstract type OptimizerTrace end


include("optimize.jl")
include("ProximalGradient.jl")



export ProximalGradient, ProximalGradientState
export VanillaProxGrad, VanillaProxGradState
export AcceleratedProxGrad, AcceleratedProxGradState


export optimize!

end
