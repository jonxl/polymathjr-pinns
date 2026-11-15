# =============================================================================
# Test: variants module
# =============================================================================
# Verifies:
#   1. VariantSpec construction + validation
#   2. build_jobs round-robin device assignment
#   3. device_names() with and without CUDA
#   4. generate_shared_dataset is deterministic and well-formed
#   5. write_manifest writes a parseable JSON
#   6. End-to-end: a single tiny variant actually trains and produces a
#      checkpoint + GPUBoard slot update
# =============================================================================

using Test
using JSON
using Random

include("../utils/plugboard.jl")
using .Plugboard

include("../utils/tui.jl")
using .TUI

include("../utils/variants.jl")
using .Variants

# ---------------------------------------------------------------------------
# Test 1: VariantSpec construction + validation
# ---------------------------------------------------------------------------

@testset "VariantSpec: construction + representation validation" begin
  spec = VariantSpec("ps_N20", :power_series;
                     N=20, neuron_count=64, maxiters=3000, seed_offset=1)
  @test spec.name == "ps_N20"
  @test spec.representation === :power_series
  @test spec.N == 20
  @test spec.neuron_count == 64
  @test spec.maxiters == 3000
  @test spec.seed_offset == 1

  # Invalid representation raises
  @test_throws ErrorException VariantSpec("bad", :invalid_rep)
end

# ---------------------------------------------------------------------------
# Test 2: build_jobs round-robin device assignment
# ---------------------------------------------------------------------------

@testset "build_jobs: round-robin device assignment" begin
  dataset = Dict{Matrix{Float32}, Vector{Float32}}()
  specs = [VariantSpec("a", :power_series; N=20),
           VariantSpec("b", :power_series; N=25),
           VariantSpec("c", :power_series; N=30),
           VariantSpec("d", :power_series; N=35),
           VariantSpec("e", :power_series; N=40)]

  # 3 devices, 5 jobs → 1, 2, 3, 1, 2
  jobs = build_jobs(specs, dataset, "results/test"; n_devices=3)
  @test length(jobs) == 5
  @test [j.device_id for j in jobs] == [1, 2, 3, 1, 2]

  # 1 device → all device_id = 1
  jobs1 = build_jobs(specs, dataset, "results/test"; n_devices=1)
  @test all(j -> j.device_id == 1, jobs1)

  # Output dirs are unique per tag
  @test length(unique(j.output_dir for j in jobs)) == 5
end

# ---------------------------------------------------------------------------
# Test 3: device_names() — CUDA-dependent behavior
# ---------------------------------------------------------------------------

@testset "device_names: CUDA fallback" begin
  names = device_names()
  @test names isa Vector{String}
  @test !isempty(names)
  if Variants.CUDA.functional()
    @test all(n -> startswith(n, "GPU ") || n == "CPU", names)
  else
    @test names == ["CPU"]
  end
end

# ---------------------------------------------------------------------------
# Test 4: generate_shared_dataset determinism + shape
# ---------------------------------------------------------------------------

@testset "generate_shared_dataset: deterministic, well-formed" begin
  ds1 = generate_shared_dataset(n_per_region=20, num_terms=10, seed=42)
  ds2 = generate_shared_dataset(n_per_region=20, num_terms=10, seed=42)
  @test ds1 == ds2                        # same seed → identical dataset
  @test length(ds1) >= 20 * 6             # at least n_per_region × 6 regions

  # Different seed → different dataset
  ds3 = generate_shared_dataset(n_per_region=20, num_terms=10, seed=43)
  @test ds1 != ds3

  # Each value is a Float32 vector of length num_terms+1
  for (k, v) in ds1
    @test v isa AbstractVector{Float32}
    @test length(v) == 11                # num_terms + 1
  end
end

# ---------------------------------------------------------------------------
# Test 5: write_manifest produces valid JSON
# ---------------------------------------------------------------------------

@testset "write_manifest: valid JSON with jobs + dataset meta" begin
  tmp = mktempdir()
  dataset = generate_shared_dataset(n_per_region=10, num_terms=20, seed=1)
  specs = [VariantSpec("ps_N20", :power_series; N=20),
           VariantSpec("eig", :eigenvalue)]
  jobs = build_jobs(specs, dataset, tmp; n_devices=2)
  path = write_manifest(joinpath(tmp, "manifest.json"), jobs,
                        Dict("n_per_region" => 10, "num_terms" => 10);
                        n_devices=2)

  @test isfile(path)
  parsed = JSON.parsefile(path)
  @test haskey(parsed, "jobs")
  @test length(parsed["jobs"]) == 2
  @test parsed["jobs"][1]["tag"] == "ps_N20"
  @test parsed["jobs"][1]["representation"] == "power_series"
  @test parsed["jobs"][1]["device_id"] == 1
  @test parsed["jobs"][2]["tag"] == "eig"
  @test parsed["jobs"][2]["device_id"] == 2
  @test parsed["dataset"]["n_per_region"] == 10
end

# ---------------------------------------------------------------------------
# Test 6: End-to-end — a single tiny variant actually trains
# ---------------------------------------------------------------------------

@testset "staged runner: end-to-end single variant" begin
  tmp = mktempdir()
  dataset = generate_shared_dataset(n_per_region=5, num_terms=5, seed=7)
  specs = [VariantSpec("tiny", :power_series; N=5, neuron_count=4,
                       maxiters=20, seed_offset=0)]
  jobs = build_jobs(specs, dataset, tmp; n_devices=1)

  # Capture logs by running in notty mode to keep CI logs clean.
  withenv("JULIA_TUI_OFF" => "1", "CI" => "1") do
    run_staged_variants(jobs)
  end

  # Job directory must exist and contain a model.checkpoint + loss.csv
  job_dir = jobs[1].output_dir
  @test isdir(job_dir)
  @test isfile(joinpath(job_dir, "model.checkpoint"))
  @test isfile(joinpath(job_dir, "loss.csv"))

  # Checkpoint should be readable
  chain, p_ca, st, meta = Variants.PINN.SafeTensorSnapshots.load_any_model(joinpath(job_dir, "model.checkpoint"))
  @test meta["representation"] == "power_series"
  @test meta["variant_name"] == "tiny"
end

# ---------------------------------------------------------------------------
# Test 7: staged grid — degrees × weightings, eigenvalue once
# ---------------------------------------------------------------------------

@testset "staged_specs: grid, names, shared seeds" begin
  specs = staged_specs(degrees=(20, 35), weightings=(:uniform, :descending))
  @test [s.name for s in specs] ==
        ["ps_N20", "ps_descending_N20", "ps_N35", "ps_descending_N35", "eig"]
  @test [s.loss.coefficient_weighting for s in specs] ==
        [:uniform, :descending, :uniform, :descending, :uniform]
  # Both weightings of one degree share the seed; they differ only in the loss.
  @test specs[1].seed_offset == specs[2].seed_offset
  @test specs[3].seed_offset == specs[4].seed_offset
  @test required_terms(specs) == 35

  # Eigenvalue contributes one variant however many degrees are staged.
  @test count(s -> s.representation === :eigenvalue,
              staged_specs(degrees=(5, 10, 15, 20))) == 1
  @test length(staged_specs(representations=(:eigenvalue,))) == 1

  @test_throws ErrorException staged_specs(degrees=(20, 20))
  @test_throws ErrorException staged_specs(degrees=(Variants.MAX_DEGREE + 1,))
  @test_throws ErrorException staged_specs(weightings=(:unknown,))
  @test_throws ErrorException staged_specs(representations=(:other,))
end

# ---------------------------------------------------------------------------
# Test 8: a dataset too short for a variant's degree is refused
# ---------------------------------------------------------------------------

@testset "build_jobs: refuses targets shorter than the degree" begin
  specs = staged_specs(degrees=(5, 8), neuron_count=4, maxiters=2)
  short = generate_shared_dataset(n_per_region=2, num_terms=5, seed=3)
  @test_throws ErrorException build_jobs(specs, short, "results/test"; n_devices=1)

  full = generate_shared_dataset(n_per_region=2, num_terms=required_terms(specs), seed=3)
  @test all(v -> length(v) == required_terms(specs) + 1, values(full))
  @test length(build_jobs(specs, full, "results/test"; n_devices=1)) == length(specs)

  # A lower degree reads a prefix of the same rows: same ODEs, same leading values.
  for (alpha, series) in short
    @test full[alpha][1:length(series)] == series
  end
end

# ---------------------------------------------------------------------------
# Test 9: run_jobs honours the per-device lane count
# ---------------------------------------------------------------------------

@testset "run_jobs: lanes bound concurrency per device" begin
  jobs = [(output_dir="job$i", device_id=1) for i in 1:6]
  for lanes in (1, 3)
    running = Threads.Atomic{Int}(0)
    peak = Threads.Atomic{Int}(0)
    Variants.RunManagement.run_jobs(jobs; devices=1, lanes=lanes) do job
      Threads.atomic_max!(peak, Threads.atomic_add!(running, 1) + 1)
      sleep(0.2)
      Threads.atomic_sub!(running, 1)
    end
    @test peak[] == lanes
  end
  @test_throws ErrorException Variants.RunManagement.run_jobs(identity, jobs; lanes=0)
end

# ---------------------------------------------------------------------------
# Test 10: End-to-end — a uniform/descending pair trains on two lanes
# ---------------------------------------------------------------------------

@testset "staged runner: two lanes, both weightings" begin
  tmp = mktempdir()
  specs = staged_specs(representations=(:power_series,), degrees=(5,),
                       neuron_count=4, maxiters=20)
  dataset = generate_shared_dataset(n_per_region=5, num_terms=required_terms(specs), seed=7)
  jobs = build_jobs(specs, dataset, tmp; n_devices=1)

  withenv("JULIA_TUI_OFF" => "1", "CI" => "1") do
    run_staged_variants(jobs; lanes=2)
  end

  weightings = String[]
  for job in jobs
    path = joinpath(job.output_dir, "model.checkpoint")
    @test isfile(path)
    @test isfile(joinpath(job.output_dir, "loss.csv"))
    _, _, _, meta = Variants.PINN.SafeTensorSnapshots.load_any_model(path)
    @test meta["variant_name"] == job.tag
    push!(weightings, meta["coefficient_weighting"])
  end
  @test weightings == ["uniform", "descending"]
  @test_throws ErrorException run_staged_variants(jobs; lanes=0)
end

@info "============================================"
@info "ALL VARIANT TESTS COMPLETED"
@info "============================================"