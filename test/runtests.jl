using StructuredSolvers
using Test
using Random

@testset "StructuredSolvers.jl" begin
    @test 1 == 1
    # Write your tests here.

    @testset "Conjugate gradient" begin
        n = 5

        M = l1Manifold(ones(n))
        x = zeros(n)

        Random.seed!(1435)
        A = rand(n, n)
        A += A'
        A += Diagonal((1-minimum(eigvals(A))) .* ones(n))
        b = A * ones(n)

        Aoperator(η) = A * η

        ## Solves A * η + b = 0
        d, j, d_type = StructuredSolvers.solve_tCG(M, x, b, Aoperator; ν=1e-13, ϵ_residual = 1e-13, maxiter=1e5, safe_projection=false, printlev=0)

        @test isapprox(d, -ones(n))
        @test d_type == :Solved
    end

end
