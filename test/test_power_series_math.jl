# Audit the derivative-coefficient output convention against independent Taylor
# differentiation.
# Run from the repository root: julia --project=. test/test_power_series_math.jl
using Test
using Random
using Zygote
using CUDA

include("../utils/loss_functions.jl")
include("../utils/plugboard.jl")
const LF = loss_functions

function audit_settings(items; N=6, K=N+1, xs=Float32[0, 0.125, 0.5, 0.75, 1])
    return (n_terms_for_power_series=N, num_supervised=K,
            num_points=length(xs), xs=xs, x_left=0f0,
            ode_matrices=Dict{Any,Any}(items), input_encoding=:coefficients)
end

# Differentiate coefficient vectors repeatedly, independently of LF's matrices.
function polynomial_derivative(a, order, x)
    nmax = length(a) - 1
    return sum(Float64(a[n+1]) * Float64(x)^(n-order) /
               Float64(factorial(big(n-order))) for n in order:nmax)
end

function reference_losses(A, items, settings)
    pde = bc = sup = 0.0
    for (b, (key, raw)) in enumerate(items)
        alpha = Float64.(vec(key)) ./ Float64(key[end])
        c = view(A, :, b)
        for x in settings.xs
            residual = sum(alpha[k+1] * polynomial_derivative(c, k, x)
                           for k in 0:length(alpha)-1)
            pde += residual^2
        end
        bc += (c[1] - raw[1])^2 + (c[2] - raw[2])^2
        for n in 0:settings.num_supervised-1
            sup += (Float64(c[n+1]) - Float64(raw[n+1]))^2
        end
    end
    B = length(items)
    return (pde / (settings.num_points * B), bc / B,
            sup / (settings.num_supervised * B))
end

@testset "Power-series mathematical audit" begin
    # e^(2x) solves u'' - 3u' + 2u = 0; cos(x) solves u'' + u = 0.
    # Nonmonic inputs also exercise canonicalization before residual evaluation.
    N = 6
    items = [reshape(Float32[4, -6, 2], 3, 1) => Float32[2^n for n in 0:N],
             reshape(Float32[3, 0, 3], 3, 1) => Float32[isodd(n) ? 0 : (-1)^(n÷2) for n in 0:N]]
    A = randn(MersenneTwister(71), Float32, N+1, length(items)) .* 0.2f0

    @testset "Scalar, cached and batched residuals and normalization" begin
        for K in (3, N+1)
            settings = audit_settings(items; N=N, K=K)
            buf = LF.precompute_batch_buffers(settings, items, false, identity)
            expected = reference_losses(A, items, settings)
            actual = LF.batched_power_series_losses(A, buf)
            @test all(isapprox.(actual, expected; rtol=2e-6, atol=1e-7))
            cached = LF.precompute_buffers(settings, false, identity)
            for (b, (key, raw)) in enumerate(items)
                scalar = LF.LossFunctionSettings(A[:, b], N,
                    Float32.(vec(key) ./ key[end]), 0f0, raw[1:2],
                    settings.xs, settings.num_points, K, raw)
                ref = reference_losses(A[:, b:b], items[b:b], settings)
                for precomputed in (nothing, cached[key])
                    values = (LF.generate_loss_pde_value(scalar; ode_buffers=precomputed),
                              LF.generate_loss_bc_value(scalar; ode_buffers=precomputed),
                              LF.generate_loss_supervised_value(scalar; ode_buffers=precomputed))
                    @test all(isapprox.(values, ref; rtol=2e-6, atol=1e-7))
                end
            end
            # Repeating samples or collocation points must not multiply the MSE.
            doubled = LF.precompute_batch_buffers(settings, vcat(items, items), false, identity)
            @test all(isapprox.(LF.batched_power_series_losses(hcat(A, A), doubled), actual; rtol=2e-6))
            twice_xs = audit_settings(items; N=N, K=K, xs=repeat(settings.xs, 2))
            repeated = LF.precompute_batch_buffers(twice_xs, items, false, identity)
            @test all(isapprox.(LF.batched_power_series_losses(A, repeated), actual; rtol=2e-6))
        end
    end

    settings = audit_settings(items; N=N)
    buf = LF.precompute_batch_buffers(settings, items, false, identity)
    objective(C, buffers) = begin
        pde, _, sup = LF.batched_power_series_losses(C, buffers)
        settings.num_supervised * pde + sup
    end
    @testset "Autodiff versus independent finite differences" begin
        gradient = only(Zygote.gradient(C -> objective(C, buf), A))
        reference_objective(C) = begin
            pde, _, sup = reference_losses(C, items, settings)
            settings.num_supervised * pde + sup
        end
        h = 1e-5
        finite_difference = similar(A, Float64)
        for i in eachindex(A)
            plus, minus = Float64.(A), Float64.(A)
            plus[i] += h
            minus[i] -= h
            finite_difference[i] = (reference_objective(plus) - reference_objective(minus)) / (2h)
        end
        @test isapprox(gradient, finite_difference; rtol=3e-6, atol=2e-6)
    end

    @testset "Factorials and known analytic coefficients" begin
        alpha = reshape(Float32[2, -3, 1], 3, 1)
        raw = Plugboard.solve_constant_coeff_series(alpha, Float32[1, 2], 21)
        @test raw == [2.0^n for n in 0:20]
        full = [alpha => Float32.(raw)]
        full_settings = audit_settings(full; N=20)
        full_buf = LF.precompute_batch_buffers(full_settings, full, false, identity)
        output = reshape(raw, :, 1)
        pde, bc, sup = LF.batched_power_series_losses(output, full_buf)
        @test pde < 1e-9 # truncation and Float32 cancellation permit nonzero residual
        @test bc == sup == 0
        @test isapprox(vec(LF.batched_reconstruct(output, full_buf)), exp.(2 .* full_settings.xs); rtol=2e-6)
        # A unit error in derivative coefficient a_6 contributes exactly 1/K
        # to the raw coefficient MSE.
        perturbed = copy(output)
        perturbed[7, 1] += 1f0
        _, _, scaled_sup = LF.batched_power_series_losses(perturbed, full_buf)
        @test isapprox(scaled_sup, 1 / 21; rtol=1e-5)
    end

    @testset "CUDA values and gradients when available" begin
        if CUDA.functional()
            CUDA.allowscalar(false)
            gpu_buf = LF.precompute_batch_buffers(settings, items, true, CUDA.cu)
            gpu_A = CUDA.cu(A)
            @test all(isapprox.(LF.batched_power_series_losses(gpu_A, gpu_buf),
                                LF.batched_power_series_losses(A, buf); rtol=2e-5, atol=2e-6))
            cpu_grad = only(Zygote.gradient(C -> objective(C, buf), A))
            gpu_grad = only(Zygote.gradient(C -> objective(C, gpu_buf), gpu_A))
            @test isapprox(Array(gpu_grad), cpu_grad; rtol=2e-5, atol=2e-6)
        else
            @test_skip CUDA.functional()
        end
    end
end
