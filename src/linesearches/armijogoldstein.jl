@with_kw struct ArmijoGoldstein <: ManifoldLinesearch
    ω₁::Float64 = 1e-4
    ω₂::Float64 = 0.99
    τₑ::Float64 = 1.5        # expansion factor when interval is unbounded
    maxit::Int = 50
end


function linesearch(ls::ArmijoGoldstein, state, linesearchstate, pb::CompositeProblem, M::Manifold, x, ∇fₘ, d; hist=Dict())
    @unpack ω₁, ω₂, τₑ = ls

    # TODO: replace 0 by -ϵ, ϵ>0 for descent dirction criterion.
    # TODO: extrapolation step for gradient...
    if inner(M, x, ∇fₘ, d) / (norm(M, x, ∇fₘ)*norm(M, x, d)) > 0
        @warn "ArmijoGoldstein: non negative direction provided, taking opposite direction." inner(M, x, ∇fₘ, d), norm(M, x, ∇fₘ), norm(M, x, d)
        d = -d
    end

    α = 1
    α_low, α_up = 0, Inf

    F_x = F(pb, x)
    F_cand = Inf
    x_cand = deepcopy(x)
    dh_0 = inner(M, x, ∇fₘ, d)


    it_ls = 0
    validpoint = false
    while !validpoint
        x_cand = retract(M, x, α*d); state.ncalls_retr += 1
        F_cand = F(pb, x_cand); state.ncalls_f += 1; state.ncalls_g += 1

        if F_x > F_cand > F_x - 3*eps(F_cand)
            @warn "Linesearch: reached function conditionning of funtion here, #it ls: $it_ls"
            # @printf "F_x        : %.16e\n" F_x
            # @printf "F_cand     : %.16e\n" F_cand
            # @printf "eps(F_cand): %.16e\n" eps(F_cand)
            break
        end

        if (F_x + ω₁*α*dh_0 ≥ F_cand) && (F_cand ≥ F_x + ω₂*α*dh_0)
            # α = 1 should be accepted for superlinear convergence.
            validpoint = true
            break
        end

        if F_x + ω₁*α*dh_0 < F_cand
            α_up = α

            α = α_up*0.1 + α_low*0.9    # NOTE : Might do interpolation here?
        elseif F_cand < F_x + ω₂*α*dh_0
            α_low = α

            if isinf(α_up)
                α = τₑ * α_low
            else
                α = α_up*0.9 + α_low*0.1
            end
        else
            validpoint = true
        end

        it_ls += 1
        (it_ls > ls.maxit) && (break)
    end

    if F_cand > F_x + ω₁*α*dh_0
        @debug "Linesearch: no suficient decrease" F_cand F_x + ω₁*α*dh_0
    end
    if F_x + ω₂*α*dh_0 > F_cand
        @debug "Linesearch: too small step" F_x + ω₂*α*dh_0 F_cand
    end

    state.niter_manls += it_ls
    hist[:niter] = it_ls
    hist[:end_fval] = F_cand

    return x_cand
end
