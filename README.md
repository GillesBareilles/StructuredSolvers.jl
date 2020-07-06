# StructuredSolvers

Implementation of solvers for composite optimization problems:
> *min_x f(x)+g(x)*

### Demo
``` julia
    n = 20
    pb = get_lasso(n, 12, 0.6)

    x0 = zeros(n)

    optimizer = ProximalGradient()
    @time optimize!(pb, optimizer, x0)

    optimizer = ProximalGradient(extrapolation = AcceleratedProxGrad())
    @time optimize!(pb, optimizer, x0)
```


### Implemented solvers
- Proximal Gradient (with backtracking line search)
    - Vanilla proximal gradient
    `optimizer = ProximalGradient()`
    - Accelerated proximal gradient
    `optimizer = AcceleratedProximalGradient()`

### Logging / trace functionality
Each algorithm should implement the `display_logs` function, which purpose is to display things if need be, and build an `OptimizationState` object instance. This `OptimizationState` is received by the generic `optimize!`, which adds it to the vector of past optimization states and eventually returns it.

The `OptimizationState` contains basic performance indicators. Specific ones can be added as a `NamedTuple`, whose template is provided to `optimize!` by the parameter `optimstate_extensions`, with an object like:
```julia
get_iterate(state::OptimizerState) = state.x
optimstate_extensions = [
    (
        key = :iterate,
        getvalue = get_iterate
    ),
]

trace = optimize!(pb, optimizer, x0, optimstate_extensions = optimstate_extensions)
```
where `get_iterate` takes as input an `OptimizerState`. Multiple dispatch allows to adapt the getter definition to the optimizer state if need be.

### Plotting from traces




## Notes / TODOs
- Use `StructArray` for trace ?