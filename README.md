# StructuredSolvers

Implementation of solvers for composite optimization problems:
> *min_x f(x)+g(x)*

The problems should implement the oracles for `f` and `g` described in the [`CompositeProblems.jl`](https://github.com/GillesBareilles/CompositeProblems.jl) package.

### Demo
``` julia
julia> using StructuredSolvers, CompositeProblems
julia> n, m, sparsity = 10, 8, 0.8
julia> pb = get_logit_MLE(n=n, m=m, sparsity=0.8);
julia> x0 = zeros(n);

# Solving with proximal gradient
julia> optimizer = ProximalGradient()
julia> trace = optimize!(pb, optimizer, x0);

# Solving with proximal gradient
julia> optimizer = ProximalGradient(extrapolation = AcceleratedProxGrad())
julia> trace = optimize!(pb, optimizer, x0);

# Solving with Proximal Gradient and Riemannian Truncated Newton acceleration
julia> optimizer = PartlySmoothOptimizer(manifold_update = ManTruncatedNewton())
julia> trace = optimize!(pb, optimizer, x0, optparams = OptimizerParams(
        iterations_limit = 40,
        trace_length = 40,
    ));
```


### Implemented solvers
- Proximal Gradient (with backtracking line search)
    - Vanilla proximal gradient `optimizer = ProximalGradient()`
    - Accelerated proximal gradient `optimizer = AcceleratedProximalGradient()`
- Structured solvers presented in this [research paper](https://arxiv.org/abs/2012.12936) with
    - Riemannian Newton acceleration `optimizer = PartlySmoothOptimizer(manifold_update = ManNewton())`;
    - Riemannian Truncated Newton acceleration `optimizer = PartlySmoothOptimizer(manifold_update = ManTruncatedNewton())`;

The Riemannian methods involve performing a Conjugate Gradient on the tangent space of the current iterate relative to the identified manifold, and a linesearch (backtracking from 1) to ensure descent.

### Logging / trace functionality
Each algorithm should implement the `display_logs` function, which purpose is to display things if need be, and build an `OptimizationState` object instance. This `OptimizationState` is received by the generic `optimize!`, which adds it to the vector of past optimization states and eventually returns it.

The `OptimizationState` contains basic performance indicators. Specific ones can be added as a `NamedTuple`, whose template is provided to `optimize!` by the parameter `optimstate_extensions`, with an object like:
```julia
using StructuredSolvers, CompositeProblems
get_iterate(state::OptimizerState) = state.x
optimstate_extensions = [
    (
        key = :iterate,
        getvalue = get_iterate
    ),
]

trace = optimize!(pb, optimizer, x0, optimstate_extensions = optimstate_extensions)
```
Multiple dispatch allows to adapt the getter definition to the optimizer state if need be.

### Timings

Method calls may be (roughly) timed using `TimerOutputs`. This functionality is turned on/off by calling `TimerOutputs.enable_debug_timings(StructuredSolvers)` and `TimerOutputs.disable_debug_timings(StructuredSolvers)`.


## Notes / TODOs
- Use `StructArray` for trace ?