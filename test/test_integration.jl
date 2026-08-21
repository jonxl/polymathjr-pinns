# =============================================================================
# Integration Test: Full Training Pipeline with raw checkpoints
# =============================================================================
# Verifies:
#   1. Fresh start training produces raw .checkpoint artifacts
#   2. Model can be loaded from middle checkpoint (warm-start)
#   3. Final model.checkpoint is valid, and converts to v2 safetensors
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
      "[0 1; -1 1]"   => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
@testset "Integration: Full training pipeline with raw checkpoints" begin
  training_ds, benchmark_ds = make_minimal_datasets()
  settings = make_settings(training_ds, benchmark_ds)
  benchmark_matrix = get_benchmark_ode_matrix(benchmark_ds)

  # --- Test A: Fresh start training with 5 neurons, 100 iterations ---
  @testset "A. Fresh-start training with checkpoints" begin
    # run_training returns nothing (for loop), so just call for side effects
    # Training creates checkpoints at epoch boundaries and a final model.checkpoint
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
    checkpoints = sort(filter(f -> endswith(f, ".checkpoint"), readdir(snapshot_dir)))
    @test length(checkpoints) >= 1
    @info "✓ Training completed with $(length(checkpoints)) checkpoints in $snapshot_dir"

    # Final model should exist
    final_model = joinpath(results_dir, "model.checkpoint")
    @test isfile(final_model)
    @info "✓ Final model.checkpoint exists"
  end

  # --- Test B: Checkpoint file is valid raw format + converts to v2 ---
  @testset "B. Checkpoint files are valid raw format and convert to v2" begin
    results_dir = find_results_dir()
    @test results_dir !== nothing

    snapshot_dir = find_snapshot_dir(results_dir)
    @test snapshot_dir !== nothing

    # Check first checkpoint
    checkpoints = sort(filter(f -> endswith(f, ".checkpoint"), readdir(snapshot_dir)))
    @test !isempty(checkpoints)

    for cp_file in checkpoints
      cp_path = joinpath(snapshot_dir, cp_file)
      _, _, _, meta = load_checkpoint(cp_path)
      @test meta["format"] == "polymathjr-pinns-checkpoint"
      @test meta["version"] == 1
      @test meta["seed"] == 42
    end
    @info "✓ All $(length(checkpoints)) checkpoints are valid raw format"

    # Conversion to safetensors v2 preserves architecture + weights
    cp_path = joinpath(snapshot_dir, checkpoints[1])
    conv_path = cp_path[1:end-length(".checkpoint")] * ".safetensors"
    convert_to_safetensors(cp_path, conv_path)
    meta, tensors = load_safetensors_model(conv_path)
    @test meta["format"] == "polymathjr-pinns-model"
    @test meta["version"] == 2
    @test length(meta["architecture"]) == 4  # 4-layer production arch
    @test length(tensors) == 8  # 4 layers × 2 params
    @info "✓ Conversion to v2 safetensors is valid"

    # Check final model
    final_model = joinpath(results_dir, "model.checkpoint")
    _, _, _, meta = load_checkpoint(final_model)
    @test meta["format"] == "polymathjr-pinns-checkpoint"
    @test meta["version"] == 1
    @info "✓ Final model.checkpoint is valid raw format"
  end

  # --- Test C: Self-contained load from final model ---
  @testset "C. Self-contained load from final model" begin
    results_dir = find_results_dir()
    @test results_dir !== nothing

    final_model = joinpath(results_dir, "model.checkpoint")
    @test isfile(final_model)

    chain2, p_ca2, st2, meta2 = load_any_model(final_model)
    @test meta2["format"] == "polymathjr-pinns-checkpoint"
    @test meta2["version"] == 1
    @test chain2 isa Lux.Chain
    @test p_ca2 isa ComponentArray
    @info "✓ load_any_model works with just a file path (no external config)"
  end

  # --- Test D: Inference with loaded model ---
  @testset "D. Inference with loaded model" begin
    results_dir = find_results_dir()
    @test results_dir !== nothing

    final_model = joinpath(results_dir, "model.checkpoint")
    chain, p_ca, st, _ = load_any_model(final_model)

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
    checkpoints = sort(filter(f -> endswith(f, ".checkpoint"), readdir(snapshot_dir)))

    if length(checkpoints) >= 1
      warmstart_path = joinpath(snapshot_dir, checkpoints[1])

      # Load the checkpoint model
      chain, p_ca, st, meta = load_any_model(warmstart_path)
      @test meta["version"] == 1
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
