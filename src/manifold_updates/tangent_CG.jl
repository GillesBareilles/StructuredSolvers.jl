"""
dₖ = solve_tCG(M, x, gradf_k, hessf_x_h, ν=1e-2, ϵ_residual = 1e-5)

Conjugate gradient method to solve the linear problem:
find d s.t. hessf_x_h(d) + gradf_k = 0

Note that we are looking for a direction d in the tangent space of M at x, and the kernel of
∇²f_M contains at least the orthogonal to this set. CG should naturally give a solution
living in the tangent space.
"""
function solve_tCG(M, x, gradfₖ, hessf_x_h; ν=1e-3, ϵ_residual = 1e-13, maxiter=1e5, safe_projection=false, printlev=0)
    norm_x(η) = norm(M, x, η)

    dₖ = zero_tangent_vector(M, x)
    rⱼ, rⱼ_prev = zero_tangent_vector(M, x), zero_tangent_vector(M, x)
    vⱼ, vⱼ_prev = zero_tangent_vector(M, x), zero_tangent_vector(M, x)

    d_type = :Unsolved
    j = 0

    (printlev>0) && @printf "\nj     norm(rⱼ)             norm(vⱼ)             ⟨vⱼ, hessf(x)[vⱼ]⟩   ν * norm(vⱼ)^2\n"
    while true
        # current residual, conjugated direction
        rⱼ = hessf_x_h(dₖ) + gradfₖ
        if safe_projection
            rⱼ = project(M, x, rⱼ)
        end

        vⱼ = - rⱼ
        if j ≥ 1
            βⱼ = norm_x(rⱼ)^2 / norm_x(rⱼ_prev)^2
            vⱼ += βⱼ * vⱼ_prev
        end

        hessf_x_vⱼ = hessf_x_h(vⱼ)
        (printlev>0) && @printf "%5i %.10e    %.10e    % .10e     %.10e\n" j norm_x(rⱼ) norm_x(vⱼ) inner(M, x, vⱼ, hessf_x_vⱼ) ν * norm_x(vⱼ)^2
        if norm_x(rⱼ) < ϵ_residual || inner(M, x, vⱼ, hessf_x_vⱼ) < ν * norm_x(vⱼ)^2 || j > maxiter
            ## Satisfying point obtained
            if j == 0
                dₖ = -gradfₖ
            end

            if norm_x(rⱼ) < ϵ_residual
                d_type = :Solved
                # printstyled("Exiting: ||rⱼ|| < ϵ : $(norm_x(rⱼ)) < $ϵ_residual\n", color=:red)
            elseif j > maxiter
                d_type = :MaxIter
                # printstyled("Exiting: j > maxiter : $j > $maxiter\n", color=:red)
            else
                d_type = :QuasiNegCurvature
                a = inner(M, x, vⱼ, hessf_x_vⱼ)
                b = ν * norm_x(vⱼ)^2
                # printstyled("Exiting: ⟨vⱼ, hessf_x(vⱼ)⟩ < ν * ||vⱼ||² : $a < $b\n", color=:red)
            end

            break
        end

        tⱼ = - inner(M, x, rⱼ, vⱼ) / inner(M, x, vⱼ, hessf_x_vⱼ)
        dₖ += tⱼ * vⱼ

        rⱼ_prev = deepcopy(rⱼ)
        vⱼ_prev = deepcopy(vⱼ)

        j += 1
    end

    return dₖ, j, d_type
end
