# =============================================================================
# Test: Held-Out Benchmark Guarantee + Coefficient Range
# =============================================================================
# Verifies:
#   1. Generating the benchmark first and passing its matrix keys as
#      exclude_matrix_keys keeps the benchmark ODE out of the training set,
#      even when the training draw saturates the matrix space
#      (coeff_bound=10 → ~400 possible 2x1 matrices).
#   2. Control: without exclusion, a saturating draw DOES collide with the
#      benchmark — proving the test is meaningful.
#   3. Wide default bound (coeff_bound=1000): draws are nearly all unique,
#      entries are nonzero integers within the bound, and |α₀/α₁| ≤ 10.
# =============================================================================

using Test
using JSON

# Load the module under test (relative to this test file)
include("../utils/plugboard.jl")
using .Plugboard

matrix_keys(dataset) = Set{String}(k for inner in values(dataset) for k in keys(inner))

@testset "Held-out benchmark: exclusion keeps benchmark out of training" begin
  bench_path = tempname() * ".json"
  train_path = tempname() * ".json"
  try
    # Benchmark first (1 ODE), small bound to make the space saturable
    bench_settings = Plugboard.Settings(1, 0, 1, bench_path, 10)
    Plugboard.generate_random_ode_dataset(bench_settings, 1; coeff_bound=10)
    held_out = matrix_keys(JSON.parsefile(bench_path))
    @test length(held_out) == 1

    # Saturating training draw (5000 draws over ~400 possible 2x1 matrices)
    train_settings = Plugboard.Settings(1, 0, 5000, train_path, 10)
    Plugboard.generate_random_ode_dataset(train_settings, 1; exclude_matrix_keys=held_out, coeff_bound=10)
    train_keys = matrix_keys(JSON.parsefile(train_path))

    # The guarantee: zero overlap despite saturation
    @test isempty(intersect(train_keys, held_out))
    # Sanity: the draw really did saturate most of the space
    @test length(train_keys) > 300

    @info "✓ Held-out guarantee holds at saturation ($(length(train_keys)) unique training matrices)"
  finally
    rm(bench_path; force=true)
    rm(train_path; force=true)
  end
end

@testset "Control: without exclusion a saturating draw collides" begin
  bench_path = tempname() * ".json"
  ctrl_path = tempname() * ".json"
  try
    bench_settings = Plugboard.Settings(1, 0, 1, bench_path, 10)
    Plugboard.generate_random_ode_dataset(bench_settings, 1; coeff_bound=10)
    held_out = matrix_keys(JSON.parsefile(bench_path))

    ctrl_settings = Plugboard.Settings(1, 0, 5000, ctrl_path, 10)
    Plugboard.generate_random_ode_dataset(ctrl_settings, 1; coeff_bound=10)  # no exclusion
    ctrl_keys = matrix_keys(JSON.parsefile(ctrl_path))

    # With ~400 possible matrices and 5000 draws, collision is a near-certainty.
    @test !isempty(intersect(ctrl_keys, held_out))

    @info "✓ Control confirmed: no exclusion → benchmark appears in training set"
  finally
    rm(bench_path; force=true)
    rm(ctrl_path; force=true)
  end
end

@testset "Wide default bound: large space, nonzero entries, ratio cap" begin
  n_draws = 2000
  keys_seen = Set{String}()
  for _ in 1:n_draws
    m = Plugboard.generate_random_alpha_matrix_with_constraint(1, 0)
    α0, α1 = m[1, 1], m[2, 1]
    @test α0 != 0 && α1 != 0
    @test abs(α0) <= 1000 && abs(α1) <= 1000
    @test abs(α0) <= 10 * abs(α1)   # ratio cap: |α₀/α₁| ≤ 10
    push!(keys_seen, string(m))
  end
  # In a ~2M matrix space, 2000 draws should be nearly collision-free
  @test length(keys_seen) > 1950

  @info "✓ Wide bound verified: $(length(keys_seen))/$n_draws unique matrices, all within bound and ratio cap"
end
