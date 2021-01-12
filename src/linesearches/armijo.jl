@with_kw struct Armijo{T} <: ManifoldLinesearch
    α::Float64 = 1e-4
    maxit::Int = 50
    initstep_strat::T = UnitInitStep()
end

@with_kw mutable struct ArmijoState <: AbstractUpdateState
    cur_funval::Float64 = Inf
    prev_funval::Float64 = Inf
    prev_initslope::Float64 = Inf
end

initial_state(ls::Armijo) = ArmijoState()


#
### Unit step size selection strategies
#
abstract type UnitStepsizeStrategy end
struct UnitInitStep <: UnitStepsizeStrategy end
# struct QuadInterpolationInitStep <: UnitStepsizeStrategy end

function init_stepsize(::UnitInitStep, lsstate::ArmijoState)
    return 1.0
end

# function init_stepsize(::QuadInterpolationInitStep, lsstate::ArmijoState)
#     α₀ = 1.0
#     if isinf(lsstate.cur_funval) || isinf(lsstate.prev_funval) || isinf(lsstate.prev_initslope)
#         @warn "LS: infinite value " isinf(lsstate.cur_funval) isinf(lsstate.prev_funval) isinf(lsstate.prev_initslope)
#     else
#         α₀ = 2*(lsstate.cur_funval - lsstate.prev_funval) / lsstate.prev_initslope
#     end
#     return min(1.01α₀, 1.0)
# end



"""
linesearch(ls::Armijo, state, pb::CompositeProblem, M::Manifold, x, ∇fₘ, d; hist=Dict())

Performs a backtracking linesearch following algorithm A.6.3.1 from [p. 326, Dennis Schnabel 1996]
"""
function linesearch(ls::Armijo, optimizerstate, linesearchstate, pb::CompositeProblem, M::Manifold, x, ∇fₘ, d; hist=Dict())
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

    F_x = F(pb, x)
    F_cand = Inf
    F_candprev = Inf
    x_cand = deepcopy(x)
    # dh_0 = inner(M, x, ∇fₘ, d)

    linesearchstate.cur_funval = F_x
    λ::Float64 = init_stepsize(ls.initstep_strat, linesearchstate)
    λprev::Float64 = deepcopy(λ)

    it_ls = 0
    while retcode == :IncompleteExecution
        x_cand = retract(M, x, λ*d); optimizerstate.ncalls_retr += 1
        F_cand = F(pb, x_cand); optimizerstate.ncalls_f += 1; optimizerstate.ncalls_g += 1

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
        elseif λ < minlambda
            retcode = :Failure
            @warn "Armijo linesearch: failure"
            x_cand = retract(M, x, 0.0*d)
            break
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

    optimizerstate.niter_manls += it_ls
    hist[:niter] = it_ls
    hist[:retcode] = retcode
    hist[:maxtaken] = maxtaken
    # hist[:end_fval] = F_cand


    ## Update linesearch state
    linesearchstate.prev_funval = F_x
    linesearchstate.prev_initslope = initslope

    return x_cand
end
