@with_kw struct ArmijoGoldstein <: ManifoldLinesearch
    ω₁::Float64 = 1e-4
    ω₂::Float64 = 0.99
    τₑ::Float64 = 1.5        # expansion factor when interval is unbounded
    maxit::Int = 50
end


function linesearch(ls::ArmijoGoldstein, pb::CompositeProblem, M::Manifold, x, ∇fₘ, d; hist=Dict())
    @unpack ω₁, ω₂, τₑ = ls

    # TODO: replace 0 by -ϵ, ϵ>0 for descent dirction criterion.
    # TODO: extrapolation step for gradient...
    @assert inner(M, x, ∇fₘ, d) / (norm(M, x, ∇fₘ)*norm(M, x, d)) < 0

    α = 1
    α_low, α_up = 0, Inf

    F_x = F(pb, x)
    F_cand = Inf
    x_cand = deepcopy(x)
    dh_0 = inner(M, x, ∇fₘ, d)

    hist[:ncalls_f] = 1

    it_ls = 0
    validpoint = false
    while !validpoint
        x_cand = retract(M, x, α*d)
        F_cand = F(pb, x_cand)
        hist[:ncalls_f] += 1

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

    hist[:niter] = it_ls
    hist[:end_fval] = F_cand

    return x_cand
end
