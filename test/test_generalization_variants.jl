# =============================================================================
# Test: generalization variants module
# =============================================================================
# Verifies:
#   1. GeneralizationSpec construction + name/representation validation
#   2. default_generalization_specs cross-product (reps × experiments)
#   3. build_gen_jobs round-robin device assignment
#   4. build_gen_jobs path uses "-" as a separator, not a literal directory
#   5. End-to-end: a tiny showcase run produces a PanelSet + manifest dir
#      for both :power_series and :eigenvalue in the same dispatch
# =============================================================================

using Test
using JSON

include("../utils/tui.jl")
using .TUI

include("../utils/generalization_variants.jl")
using .GeneralizationVariants

# ---------------------------------------------------------------------------
# Test 1: GeneralizationSpec construction + validation
# ---------------------------------------------------------------------------

@testset "GeneralizationSpec: construction + validation" begin
  spec = GeneralizationSpec(:transfer, :power_series;
                             regions=[:saddle, :stable_node],
                             Rmax=4, Ng=21,
                             n_per_region=100, maxiters=2000)
  @test spec.name === :transfer
  @test spec.representation === :power_series
  @test spec.regions == [:saddle, :stable_node]
  @test spec.Rmax == 4
  @test spec.Ng == 21
  @test spec.n_per_region == 100
  @test spec.maxiters == 2000

  # Invalid name raises
  @test_throws ErrorException GeneralizationSpec(:bogus, :power_series)

  # Invalid representation raises
  @test_throws ErrorException GeneralizationSpec(:transfer, :unsupported_rep)
end

# ---------------------------------------------------------------------------
# Test 2: default_generalization_specs cross-product
# ---------------------------------------------------------------------------

@testset "default_generalization_specs: cross-product" begin
  specs = default_generalization_specs()
  # 2 reps × 5 experiments = 10 specs by default
  @test length(specs) == 10

  # Custom subset
  specs2 = default_generalization_specs(
    reps=(:power_series,),
    experiments=(:transfer, :extrapolate),
  )
  @test length(specs2) == 2
  @test all(s -> s.representation === :power_series, specs2)
  @test Set(s.name for s in specs2) == Set([:transfer, :extrapolate])

  # kwargs forwarded
  specs3 = default_generalization_specs(
    reps=(:eigenvalue,),
    experiments=(:gen_radius,),
    Rmax=3, Ng=11, n_per_region=50, maxiters=1000,
  )
  @test length(specs3) == 1
  @test specs3[1].Rmax == 3
  @test specs3[1].Ng == 11
  @test specs3[1].n_per_region == 50
  @test specs3[1].maxiters == 1000
end

# ---------------------------------------------------------------------------
# Test 3: build_gen_jobs round-robin
# ---------------------------------------------------------------------------

@testset "build_gen_jobs: round-robin device assignment" begin
  specs = default_generalization_specs(reps=(:power_series,),
                                       experiments=(:transfer, :extrapolate,
                                                    :range, :showcase, :gen_radius))

  # 2 devices, 5 jobs → 1, 2, 1, 2, 1
  jobs = build_gen_jobs(specs, "/tmp/x"; n_devices=2)
  @test length(jobs) == 5
  @test [j.device_id for j in jobs] == [1, 2, 1, 2, 1]

  # 1 device → all device_id = 1
  jobs1 = build_gen_jobs(specs, "/tmp/x"; n_devices=1)
  @test all(j -> j.device_id == 1, jobs1)

  # Output dirs are unique
  @test length(unique(j.output_dir for j in jobs)) == 5
end

# ---------------------------------------------------------------------------
# Test 4: build_gen_jobs path separator
# ---------------------------------------------------------------------------

@testset "build_gen_jobs: '-' stays a separator in output path" begin
  specs = [GeneralizationSpec(:transfer, :power_series)]
  jobs = build_gen_jobs(specs, "/tmp/x"; n_devices=1)
  @test jobs[1].output_dir == "/tmp/x/gen-transfer-power_series"

  # Make sure no literal "-" directory
  @test !occursin("/-/p", jobs[1].output_dir)
end

# ---------------------------------------------------------------------------
# Test 5: End-to-end — one rep, one experiment, tiny params
# ---------------------------------------------------------------------------

@testset "generalization variants: end-to-end showcase/power_series" begin
  tmp = mktempdir()
  specs = [GeneralizationSpec(:showcase, :power_series;
                              n_per_region=3, maxiters=10)]
  jobs = build_gen_jobs(specs, tmp; n_devices=1)

  withenv("JULIA_TUI_OFF" => "1", "CI" => "1") do
    run_generalization_variants(jobs)
  end

  job_dir = jobs[1].output_dir
  @test isdir(job_dir)
  @test isfile(joinpath(job_dir, "panelset.json"))

  # Panelset parses as JSON
  parsed = JSON.parsefile(joinpath(job_dir, "panelset.json"))
  @test haskey(parsed, "panels")
end

# ---------------------------------------------------------------------------
# Test 6: End-to-end — both reps dispatched concurrently
# ---------------------------------------------------------------------------

@testset "generalization variants: both reps produce panelsets" begin
  tmp = mktempdir()
  specs = [
    GeneralizationSpec(:showcase, :power_series; n_per_region=3, maxiters=10),
    GeneralizationSpec(:showcase, :eigenvalue;  n_per_region=3, maxiters=10),
  ]
  jobs = build_gen_jobs(specs, tmp; n_devices=1)

  withenv("JULIA_TUI_OFF" => "1", "CI" => "1") do
    run_generalization_variants(jobs)
  end

  # Both jobs should have produced their own panelset.json
  @test isfile(joinpath(tmp, "gen-showcase-power_series", "panelset.json"))
  @test isfile(joinpath(tmp, "gen-showcase-eigenvalue",  "panelset.json"))

  ps_ps = JSON.parsefile(joinpath(tmp, "gen-showcase-power_series", "panelset.json"))
  ps_eig = JSON.parsefile(joinpath(tmp, "gen-showcase-eigenvalue",  "panelset.json"))
  @test haskey(ps_ps, "panels")
  @test haskey(ps_eig, "panels")
end

@info "============================================"
@info "ALL GENERALIZATION VARIANT TESTS COMPLETED"
@info "============================================"