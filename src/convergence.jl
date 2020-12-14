function has_converged(cvchecker::ConvergenceChecker, optimizerstate::OptimizerState)
    return false
end


@with_kw struct FirstOrderOptimality <: ConvergenceChecker
    tol_tangent::Float64 = 1e-11
    tol_normal::Float64 = 1e-11
end

function has_converged(cvchecker::FirstOrderOptimality, pb::CompositeProblem, o::Optimizer, os::OptimizationState)
    return (os.norm_minsubgradient_tangent < cvchecker.tol_tangent) && (os.norm_minsubgradient_normal < cvchecker.tol_normal)
end
