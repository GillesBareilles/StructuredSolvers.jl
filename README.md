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