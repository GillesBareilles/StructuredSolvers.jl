
"""
dₖ = solve_tCG(M, x, gradf_k, hessf_x_h, ν=1e-2, ϵ_residual = 1e-5)

Conjugate gradient method to solve the linear problem:
find d s.t. hessf_x_h(d) + gradf_k = 0

Note that we are looking for a direction d in the tangent space of M at x, and the kernel of
∇²f_M contains at least the orthogonal to this set. CG should naturally give a solution
living in the tangent space.
"""
function solve_tCG(M, x, gradfₖ, hessf_x_h; ν=1e-3, ϵ_residual = 1e-13, maxiter=1e5, safe_projection=true, printlev=0)

    dₖ = zero(gradfₖ)
    rⱼ, rⱼ_prev = zero(gradfₖ), zero(gradfₖ)
    vⱼ, vⱼ_prev = zero(gradfₖ), zero(gradfₖ)

    j = 0

    (printlev>0) && @printf "\nj     norm(rⱼ)             ⟨vⱼ, hessf(x)[vⱼ]⟩   ν * norm(vⱼ)^2\n"
    while true
        # current residual, conjugated direction
        rⱼ = hessf_x_h(dₖ) + gradfₖ
        if safe_projection
            rⱼ = project(M, x, x+rⱼ)
        end

        vⱼ = - rⱼ
        if j ≥ 1
            βⱼ = norm(rⱼ)^2 / norm(rⱼ_prev)^2
            vⱼ += βⱼ * vⱼ_prev
        end

        hessf_x_vⱼ = hessf_x_h(vⱼ)
        (printlev>0) && @printf "%5i %.10e    % .10e     %.10e\n" j norm(rⱼ) inner(M, x, vⱼ, hessf_x_vⱼ) ν * norm(vⱼ)^2
        if norm(rⱼ) < ϵ_residual || inner(M, x, vⱼ, hessf_x_vⱼ) < ν * norm(vⱼ)^2 || j > maxiter
            ## Satisfying point obtained
            if j == 0
                dₖ = -gradfₖ
            end
            break
        end

        tⱼ = - inner(M, x, rⱼ, vⱼ) / inner(M, x, vⱼ, hessf_x_vⱼ)
        dₖ += tⱼ * vⱼ

        rⱼ_prev = copy(rⱼ)
        vⱼ_prev = copy(vⱼ)

        j += 1
    end

    return dₖ, j
end
