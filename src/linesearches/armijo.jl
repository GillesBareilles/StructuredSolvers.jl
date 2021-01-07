@with_kw struct Armijo <: ManifoldLinesearch
    α::Float64 = 1e-4
    maxit::Int = 50
end

"""
linesearch(ls::Armijo, state, pb::CompositeProblem, M::Manifold, x, ∇fₘ, d; hist=Dict())

Performs a backtracking linesearch following algorithm A.6.3.1 from [p. 326, Dennis Schnabel 1996]
"""
function linesearch(ls::Armijo, state, pb::CompositeProblem, M::Manifold, x, ∇fₘ, d; hist=Dict())
    # TODO: replace 0 by -ϵ, ϵ>0 for descent dirction criterion.
    # TODO: extrapolation step for gradient...
    if inner(M, x, ∇fₘ, d) / (norm(M, x, ∇fₘ)*norm(M, x, d)) > 0
        @warn "Armijo: non negative direction provided, taking opposite direction." inner(M, x, ∇fₘ, d), norm(M, x, ∇fₘ), norm(M, x, d)
        d = -d
    end

    maxstep = 1e10

    maxtaken = false
    retcode = :IncompleteExecution
    @unpack α, maxit = ls

    newtlen = norm(M, x, d)
    if newtlen > maxstep
        d *= maxstep / newtlen
    end

    initslope = inner(M, x, ∇fₘ, d)
    # rellength = ? #What is Sₓ
    minlambda = 1e-15
    # TODO
    @debug "Armijo: rellength stop crit not implemented..."

    λ::Float64 = 1.0


    F_x = F(pb, x)
    F_cand = Inf
    F_candprev = Inf
    x_cand = deepcopy(x)
    # dh_0 = inner(M, x, ∇fₘ, d)
    λprev::Float64 = 1.0

    it_ls = 0
    while retcode == :IncompleteExecution
        x_cand = retract(M, x, λ*d); state.ncalls_retr += 1
        F_cand = F(pb, x_cand); state.ncalls_f += 1; state.ncalls_g += 1

        if F_x > F_cand > F_x - 3*eps(F_cand)
            @warn "Armijo linesearch: reached function conditionning of funtion here, #it ls: $it_ls"
            break
        end

        if F_cand ≤ F_x + α*λ*initslope
            retcode = :Satisfactory
            if λ==1 && (newtlen > 0.99*maxstep)
                maxtaken = true
            end
            break
            # return x_cand
        elseif λ < minlambda
            retcode = :Failure
            @warn "Armijo linesearch: failure"
            x_cand = retract(M, x, 0.0*d)
            break
            # return x
        else
            if λ == 1.0
                λtemp = -initslope / (2*(F_cand - F_x - initslope))
            else
                a, b = 1/(λ-λprev) * [
                    1/λ^2 -1/(λprev^2)
                    -λprev/λ^2  λ/(λprev^2)
                ] * [
                    F_cand - F_x - λ*initslope
                    F_candprev - F_x - λprev*initslope
                ]
                disc = b^2 - 3*a*initslope
                if a==0
                    λtemp = -initslope / (2b)
                else
                    λtemp = (-b+sqrt(disc)) / (3a)
                end

                λtemp = min(λtemp, 0.5λ)
            end

            λprev = λ
            F_candprev = F_cand

            λ = max(λtemp, 0.1λ)
        end

        it_ls += 1
        (it_ls > ls.maxit) && (break)
    end

    state.niter_manls += it_ls
    hist[:niter] = it_ls
    hist[:retcode] = retcode
    hist[:maxtaken] = maxtaken
    # hist[:end_fval] = F_cand

    return x_cand
end
