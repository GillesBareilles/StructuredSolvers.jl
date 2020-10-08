@with_kw struct OptimizerParams
    iterations_limit::Int64 = 30
    time_limit::Float64 = 30.0
    show_trace::Bool = true
    trace_length::Int64 = 20
    cvcheckers::Set{ConvergenceChecker} = Set{ConvergenceChecker}([ProxGradStepLength()])
end
