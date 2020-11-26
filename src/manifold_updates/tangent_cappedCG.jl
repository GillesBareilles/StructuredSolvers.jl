function update_paramsfromM(M, ϵ, ζ)
    κ = (M+2ϵ) / ϵ
    ζ̂ = ζ / (3κ)
    τ = √(κ) / (√(κ) + 1)
    T = 4 * κ^4 / (1-√(τ))^2
    return κ, ζ̂, τ, T
end

"""
dₖ = solve_tCG(M, x, gradf_k, hessf_x_h, ν=1e-2, ϵ_residual = 1e-5)

Conjugate gradient method to solve the linear problem:
find d s.t. hessf_x_h(d) + gradf_k = 0

An implementation of ``capped CG'' following
    *A Newton-CG algorithm with complexity guarantees for smooth
    unconstrained optimization*, Royer, Neil, Wright.
"""
function solve_tCG_capped(Man, x, gradfₖ, hessf_x_h; ϵ=1e-8, ζ = 1e-8, maxiter=1e5, check_tvectors=false, printlev=0)

    # Short notations
    norm_x(η) = norm(Man, x, η)
    inner_x(η, ξ) = inner(Man, x, η, ξ)
    H(η) = hessf_x_h(η)
    H̄(η) = hessf_x_h(η) + 2ϵ*η

    # Operator norm of H
    M::Float64 = 0.0

    κ, ζ̂, τ, T = update_paramsfromM(M, ϵ, ζ)

    r₀ = gradfₖ
    yⱼ = zero_tangent_vector(Man, x)
    rⱼ, rⱼ_prev = deepcopy(gradfₖ), deepcopy(gradfₖ)
    pⱼ, pⱼ_prev = deepcopy(-gradfₖ), deepcopy(-gradfₖ)
    j = 0


    # ##
    # @show ϵ
    # n = size(x, 1)
    # @show Man.nnz_coords

    # Hess = zeros(10, 10)
    # Hess_bar = zeros(10, 10)

    # for i in 1:n
    #     e_i = zeros(n)
    #     e_i[i] = 1.0

    #     Hess[:, i] = H(e_i)
    #     Hess_bar[:, i] = H̄(e_i)
    # end

    # @show (eigvals(Hess))
    # @show (eigvals(Hess_bar))

    # d = rand(10)
    # @assert norm(Hess*d - H(d)) < 1e-14
    # ##



    if inner_x(pⱼ, H̄(pⱼ)) < ϵ * inner_x(pⱼ, pⱼ)
        return pⱼ, :NegativeCurvature, j
    elseif norm_x(H(pⱼ)) > M * norm_x(pⱼ)
        M = norm_x(H(pⱼ)) / norm_x(pⱼ)
        κ, ζ̂, τ, T = update_paramsfromM(M, ϵ, ζ)
    end

    (printlev>0) && @printf "\nj     norm(rⱼ)             norm(vⱼ)             ⟨vⱼ, hessf(x)[vⱼ]⟩   ν * norm(vⱼ)^2\n"
    while true

        rⱼ_prev = deepcopy(rⱼ)
        # yⱼ_prev = deepcopy(yⱼ)
        pⱼ_prev = deepcopy(pⱼ)


        # Standard CG operations
        αⱼ = inner_x(rⱼ, rⱼ) / inner_x(pⱼ, H̄(pⱼ))
        yⱼ += αⱼ * pⱼ
        rⱼ += αⱼ * H̄(pⱼ)

        #TODO: storing the inner product is enough here
        βⱼ = inner_x(rⱼ, rⱼ) / inner_x(rⱼ_prev, rⱼ_prev)
        pⱼ = -rⱼ + βⱼ * pⱼ_prev
        j += 1

        # Update estimate of H's operator norm
        for v in [pⱼ, yⱼ, rⱼ]
            norm_Hv = norm_x(H(v))
            norm_v = norm_x(v)
            if norm_Hv > M * norm_v
                M = norm_Hv / norm_v
                κ, ζ̂, τ, T = update_paramsfromM(M, ϵ, ζ)
            end
        end


        # Plotting
        if printlev>0
            # @printf "%5i %.10e    %.10e    % .10e     %.10e\n" j norm_x(rⱼ) norm_x(vⱼ) inner(M, x, vⱼ, hessf_x_vⱼ) ν * norm_x(vⱼ)^2
            @printf "%5i %.10e\n" j norm_x(rⱼ)
        end


        # Termination criteria
        if inner_x(yⱼ, H̄(yⱼ)) < ϵ * inner_x(yⱼ, yⱼ)
            return yⱼ, :NegativeCurvature, j
        elseif norm_x(rⱼ) ≤ ζ̂ * norm_x(r₀)
            return yⱼ, :Solution, j
        elseif inner_x(pⱼ, H̄(pⱼ)) < ϵ * inner_x(pⱼ, pⱼ)
            return pⱼ, :NegativeCurvature, j
        elseif norm_x(rⱼ) > √(T) * τ^(j/2) * norm_x(r₀)
            # αⱼ = inner_x(rⱼ, rⱼ) / inner_x(pⱼ, H(pⱼ))
            # yⱼ += αⱼ * pⱼ
            # TODO : implement last item, by executing again iterations of CG (instead of storing)
            @warn "Weird unfound negative curvature direction. Not implemented."
            @assert false
        end

        if j >= 400
            return -yⱼ * sign(inner_x(yⱼ, gradfₖ)), :Solution, j
        end
        # (printlev>0) && @printf "%5i %.10e    %.10e    % .10e     %.10e\n" j norm_x(rⱼ) norm_x(vⱼ) inner(M, x, vⱼ, hessf_x_vⱼ) ν * norm_x(vⱼ)^2
        # if norm_x(rⱼ) < ϵ_residual || inner(M, x, vⱼ, hessf_x_vⱼ) < ν * norm_x(vⱼ)^2 || j > maxiter

        # j += 1
    end

    return
end
