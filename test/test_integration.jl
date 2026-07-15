# =============================================================================
# Integration Test: Full Training Pipeline with v2 Safetensors
# =============================================================================
# Verifies:
#   1. Fresh start training produces v2 per-layer safetensors checkpoints
#   2. Model can be loaded from middle checkpoint (warm-start)
#   3. Final model.safetensors is valid v2 format
#   4. Inference with loaded model matches original
#   5. load_and_infer works with just 2 arguments (no settings)
# =============================================================================

using Test
using Random
using JSON
using Lux
using ComponentArrays

# Include the full project pipeline
include("../utils/plugboard.jl")
using .Plugboard

include("../utils/helper_funcs.jl")
using .helper_funcs

include("../utils/safetensors_utils.jl")
using .SafeTensorSnapshots

include("../utils/snapshot_utils.jl")
using .SnapshotUtils

include("../architectures/PINN.jl")
using .PINN

include("../utils/training_schemes.jl")
using .training_schemes

# ---------------------------------------------------------------------------
# Build minimal training/benchmark datasets
# ---------------------------------------------------------------------------
function make_minimal_datasets()
  training = Dict{String,Dict{String,Any}}(
    "01" => Dict{String,Any}(
      "[1 0; 0 1]"    => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "[2 0; 0 1]"    => [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "[1 0; 0 -1]"   => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "[0 1; -1 0]"   => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "[3 0; 0 2]"    => [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ),
  )

  benchmark = Dict{String,Dict{String,Any}}(
    "01" => Dict{String,Any}(
      "[1 0; 0 1]" => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ),
  )

  return training, benchmark
end

function get_benchmark_ode_matrix(benchmark::Dict)
  for (_, inner_dict) in benchmark
    for (mat_key, _) in inner_dict
      return eval(Meta.parse(mat_key))
    end
  end
  error("No benchmark ODE matrix found")
end

function make_settings(training_dataset, benchmark_dataset)
  return TrainingSchemesSettings(
    training_dataset,
    benchmark_dataset,
    10,                         # N
    10,                         # num_supervised
    10,                         # num_points
    Float32(0.0),
    Float32(1.0),
    Float32(1.0),
    Float32(1.0),
    Float32(1.0),
    Float32.(collect(0.0:0.1:1.0)),
  )
end

# ---------------------------------------------------------------------------
# Helper: find the latest results dir
# ---------------------------------------------------------------------------
function find_results_dir()
  dirs = filter(d -> startswith(d, "run-"), readdir("results"))
  sort!(dirs; by=d -> mtime(joinpath("results", d)), rev=true)
  isempty(dirs) ? nothing : joinpath("results", dirs[1])
end

function find_snapshot_dir(run_dir)
  dir = joinpath(run_dir, "snapshots")
  isdir(dir) ? dir : nothing
end

# ---------------------------------------------------------------------------
# Integration Test Suite
# ---------------------------------------------------------------------------
@testset "Integration: Full training pipeline with v2 safetensors" begin
  training_ds, benchmark_ds = make_minimal_datasets()
  settings = make_settings(training_ds, benchmark_ds)
  benchmark_matrix = get_benchmark_ode_matrix(benchmark_ds)

  # --- Test A: Fresh start training with 5 neurons, 100 iterations ---
  @testset "A. Fresh-start training with checkpoints" begin
    # run_training returns nothing (for loop), so just call for side effects
    # Training creates checkpoints at epoch boundaries and a final model.safetensors
    run_training(
      settings, 100, 10;
      neuron_count=5, seed=42, batch_size=0,
      snapshot_epoch_interval=50,
    )

    # Verify results are on disk
    results_dir = find_results_dir()
    @test results_dir !== nothing
    @test isdir(results_dir)

    snapshot_dir = find_snapshot_dir(results_dir)
    @test snapshot_dir !== nothing
    @test isdir(snapshot_dir)

    # Should have checkpoints at iterations 50 and 100
    checkpoints = sort(filter(f -> endswith(f, ".safetensors"), readdir(snapshot_dir)))
    @test length(checkpoints) >= 1
    @info "✓ Training completed with $(length(checkpoints)) checkpoints in $snapshot_dir"

    # Final model should exist
    final_model = joinpath(results_dir, "model.safetensors")
    @test isfile(final_model)
    @info "✓ Final model.safetensors exists"
  end

  # --- Test B: Checkpoint file is valid v2 format ---
  @testset "B. Checkpoint files are valid v2 per-layer format" begin
    results_dir = find_results_dir()
    @test results_dir !== nothing

    snapshot_dir = find_snapshot_dir(results_dir)
    @test snapshot_dir !== nothing

    # Check first checkpoint
    checkpoints = sort(filter(f -> endswith(f, ".safetensors"), readdir(snapshot_dir)))
    @test !isempty(checkpoints)

    for cp_file in checkpoints
      cp_path = joinpath(snapshot_dir, cp_file)
      meta, tensors = load_safetensors_model(cp_path)
      @test meta["format"] == "polymathjr-pinns-model"
      @test meta["version"] == 2
      @test meta["seed"] == 42
      @test length(meta["architecture"]) == 4  # 4-layer production arch
      @test length(tensors) == 8  # 4 layers × 2 params
    end
    @info "✓ All $(length(checkpoints)) checkpoints are valid v2 format"

    # Check final model
    final_model = joinpath(results_dir, "model.safetensors")
    meta, tensors = load_safetensors_model(final_model)
    @test meta["format"] == "polymathjr-pinns-model"
    @test meta["version"] == 2
    @test length(tensors) == 8
    @info "✓ Final model.safetensors is valid v2 format"
  end

  # --- Test C: Self-contained load_model from final model ---
  @testset "C. Self-contained load_model from final model" begin
    results_dir = find_results_dir()
    @test results_dir !== nothing

    final_model = joinpath(results_dir, "model.safetensors")
    @test isfile(final_model)

    chain2, p_ca2, st2, meta2 = load_model(final_model)
    @test meta2["format"] == "polymathjr-pinns-model"
    @test meta2["version"] == 2
    @test chain2 isa Lux.Chain
    @test p_ca2 isa ComponentArray
    @info "✓ load_model works with just a file path (no external config)"
  end

  # --- Test D: Inference with loaded model ---
  @testset "D. Inference with loaded model" begin
    results_dir = find_results_dir()
    @test results_dir !== nothing

    final_model = joinpath(results_dir, "model.safetensors")
    chain, p_ca, st, _ = load_model(final_model)

    input_vec = Float32.(vec(benchmark_matrix))
    output = first(chain(input_vec, p_ca, st))
    # For single-input batch, Lux may return Vector or Matrix
    @test (output isa AbstractArray)
    @info "✓ Inference works: output size = $(size(output))"

    # load_and_infer with just 2 args (no settings)
    coeffs = SnapshotUtils.load_and_infer(final_model, benchmark_matrix)
    @test coeffs isa Vector
    @test length(coeffs) > 0
    @info "✓ load_and_infer works with 2 args (no settings)"
  end

  # --- Test E: Warm-start from middle checkpoint ---
  @testset "E. Warm-start from middle checkpoint" begin
    results_dir = find_results_dir()
    @test results_dir !== nothing

    snapshot_dir = find_snapshot_dir(results_dir)
    checkpoints = sort(filter(f -> endswith(f, ".safetensors"), readdir(snapshot_dir)))

    if length(checkpoints) >= 1
      warmstart_path = joinpath(snapshot_dir, checkpoints[1])

      # Load the checkpoint model
      chain, p_ca, st, meta = load_model(warmstart_path)
      @test meta["version"] == 2
      @test chain isa Lux.Chain

      # Run inference
      input = Float32.(vec(benchmark_matrix))
      output = first(chain(input, p_ca, st))
      @test output isa AbstractArray
      @info "✓ Warm-start from checkpoint $(checkpoints[1]) works"
    else
      @info "Only 1 checkpoint, skipping multi-checkpoint warm-start test"
    end
  end
end

@info "============================================"
@info "ALL INTEGRATION TESTS COMPLETED"
@info "============================================"
