using Test

include("assemble.jl")

exact_poly_integral(k, a, b) = (b^(k+1) - a^(k+1)) / (k + 1)

@testset "GLL Quadrature Tests" begin

    element = [-2.0, 3.0]
    atol = 1e-8
    orders = [1, 3, 5, 7, 9]

    for p in orders
        @testset "Polynomial exactness for p = $p" begin
            for k in 0:p
                
                f(x) = x^k

                numerical = gauss_integrate(element, p, 1, f)
                exact = exact_poly_integral(k, element[1], element[2])
                @test isapprox(numerical, exact; atol=atol, rtol=0)
            end
        end
    end

    @testset "Product of functions" begin
        
        f(x) = x
        g(x) = x^2

        numerical = gauss_integrate(element, 3, 1, f, g)
        exact = exact_poly_integral(3, element[1], element[2])

        @test isapprox(numerical, exact; atol=atol)
    end

    @testset "Non-polynomial convergence behavior" begin
        
        f(x) = exp(x)
        exact = exp(element[2]) - exp(element[1])

        errors = Dict()
        for p in orders
            approx = gauss_integrate(element, p, 1, f)
            errors[p] = abs(approx - exact)
        end

        for i in 2:length(orders)
            @test errors[orders[i]] < errors[orders[i-1]]
        end
    end

    @testset "Symmetry sanity check" begin
        
        element_sym = [-1.0, 1.0]
        f(x) = x^3

        val = gauss_integrate(element_sym, 5, 1, f)
        @test isapprox(val, 0.0; atol=atol)
    end
end
