# Large, matched dual-representation pretraining run.
#
# This script trains reusable frozen models. Generalization experiments load
# these checkpoints later and perform inference; they do not retrain them.

using Dates
using JSON
using CUDA
using Zygote
using ProgressMeter

include("../../architectures/PINN.jl")
using .PINN
include("../../utils/dual_dataset.jl")
using .DualDataset
include("../../utils/tui.jl")
using .TUI

const REGIONS = [:saddle, :stable_node, :unstable_node,
                 :stable_spiral, :unstable_spiral, :center]

const DATASET_SEED = parse(Int, get(ENV, "DUAL_DATASET_SEED", "10001"))
const SHUFFLE_SEED = parse(Int, get(ENV, "DUAL_SHUFFLE_SEED", "40001"))
const MODEL_SEED = parse(Int, get(ENV, "DUAL_MODEL_SEED", "1234"))
const RUN_NAME = "dataset-$(DATASET_SEED)-shuffle-$(SHUFFLE_SEED)-model-$(MODEL_SEED)"
const OUTPUT_ROOT = joinpath("results/dual-representation-10m", RUN_NAME)
const TRAIN_SIZE = 10_000_000
const VALIDATION_SIZE = 500_000
const TEST_SIZE = 1_000_000
const BATCH_SIZE = 16_384
const EPOCHS = 100
const CHECKPOINT_EPOCH_INTERVAL = 5
const N = 20
const NEURON_COUNT = 64
const NUM_POINTS = 50
const TRAIN_SEED = DATASET_SEED
const VALIDATION_SEED = DATASET_SEED + 10_000
const TEST_SEED = DATASET_SEED + 20_000
const RUN_GLOBAL_MODELS = true
const RUN_FAMILY_MODELS = true
const GPU_COUNT = 8

# Live display backend: "progress" (ProgressMeter, one bar per GPU, offset-
# stacked) or "tui" (the hand-rolled TUI.GPUBoard). Select with
# DUAL_DISPLAY=tui julia ... scripts/shared/run_all.jl
const DISPLAY_MODE = get(ENV, "DUAL_DISPLAY", "progress")
DISPLAY_MODE in ("tui", "progress") ||
  error("DUAL_DISPLAY must be \"tui\" or \"progress\", got $(repr(DISPLAY_MODE))")

const TRAIN_SPLIT = CanonicalODESplit(:train, TRAIN_SIZE, TRAIN_SEED, N;
  shuffle_seed=SHUFFLE_SEED, regions=REGIONS)
const VALIDATION_SPLIT = CanonicalODESplit(:validation, VALIDATION_SIZE, VALIDATION_SEED, N;
  shuffle_seed=SHUFFLE_SEED + 10_000, regions=REGIONS)
const TEST_SPLIT = CanonicalODESplit(:test, TEST_SIZE, TEST_SEED, N;
  shuffle_seed=SHUFFLE_SEED + 20_000, regions=REGIONS)

split_manifest(split) = Dict(
  "name" => String(split.name),
  "size" => split.size,
  "seed" => string(split.seed),
  "shuffle_seed" => string(split.shuffle_seed),
  "dataset_id" => dataset_id(split),
  "regions" => String.(split.regions),
  "power_series_degree" => split.N,
  "materialization" => "deterministic streaming by canonical index",
)

function run_config()
  global_batches = batches_per_epoch(TRAIN_SPLIT, BATCH_SIZE)
  Dict(
    "study" => "dual-representation",
    "created_at" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
    "output_root" => OUTPUT_ROOT,
    "representations" => ["power_series", "eigenvalue"],
    "training" => Dict(
      "epochs" => EPOCHS,
      "batch_size" => BATCH_SIZE,
      "global_batches_per_epoch" => global_batches,
      "global_optimizer_updates" => global_batches * EPOCHS,
      "checkpoint_every_epochs" => CHECKPOINT_EPOCH_INTERVAL,
      "optimizer" => "Adam",
      "learning_rate" => 0.001,
      "network_hidden_width" => NEURON_COUNT,
      "collocation_points" => NUM_POINTS,
      "power_series_degree" => N,
      "model_seed" => MODEL_SEED,
      "dataset_seed" => DATASET_SEED,
      "shuffle_seed" => SHUFFLE_SEED,
      "global_models" => RUN_GLOBAL_MODELS,
      "family_models" => RUN_FAMILY_MODELS,
    ),
    "splits" => Dict(
      "train" => split_manifest(TRAIN_SPLIT),
      "validation" => split_manifest(VALIDATION_SPLIT),
      "test" => split_manifest(TEST_SPLIT),
    ),
    "experimental_control" => "Both representations use identical canonical ODE indices, split seeds, epoch permutations, and minibatches.",
    "test_policy" => "The final test split is not used for training, checkpoint selection, or hyperparameter selection.",
  )
end

function write_run_manifests()
  mkpath(OUTPUT_ROOT)
  mkpath(joinpath(OUTPUT_ROOT, "datasets"))
  cfg = run_config()
  open(joinpath(OUTPUT_ROOT, "config.json"), "w") do io
    JSON.print(io, cfg, 2)
  end
  for split in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
    open(joinpath(OUTPUT_ROOT, "datasets", "$(split.name)-manifest.json"), "w") do io
      JSON.print(io, split_manifest(split), 2)
    end
  end
  open(joinpath(OUTPUT_ROOT, "config.md"), "w") do io
    println(io, "# Dual-representation 10M training configuration\n")
    println(io, "Both representations train on the same canonical ODE indices and minibatches. The power-series and eigenvalue targets are derived from each shared ODE; they are not independently sampled datasets.\n")
    println(io, "- Training ODEs: **$(TRAIN_SIZE)**")
    println(io, "- Validation ODEs: **$(VALIDATION_SIZE)**")
    println(io, "- Final untouched test ODEs: **$(TEST_SIZE)**")
    println(io, "- Batch size: **$(BATCH_SIZE)**")
    println(io, "- Epochs: **$(EPOCHS)**")
    println(io, "- Global updates per epoch: **$(batches_per_epoch(TRAIN_SPLIT, BATCH_SIZE))**")
    println(io, "- Global total optimizer updates: **$(batches_per_epoch(TRAIN_SPLIT, BATCH_SIZE) * EPOCHS)**")
    println(io, "- Checkpoints: every **$(CHECKPOINT_EPOCH_INTERVAL)** epochs; the final model is saved separately")
    println(io, "- Power-series degree: **N=$(N)** ($(N + 1) outputs)")
    println(io, "- Eigenvalue outputs: **4**")
    println(io, "- Models: **2 global + 12 family-specific = 14 total**")
    println(io, "- Dense MLPs: power series `2→64→64→64→21` (**9,877 parameters**); eigenvalue `2→64→64→64→4` (**8,772 parameters**)")
    println(io, "- Dataset seed: **$(DATASET_SEED)**")
    println(io, "- Shuffle seed: **$(SHUFFLE_SEED)**")
    println(io, "- Model initialization seed: **$(MODEL_SEED)**")
    println(io, "- Final power-series filename: `model-dense-mlp-p009877.checkpoint`")
    println(io, "- Final eigenvalue filename: `model-dense-mlp-p008772.checkpoint`")
    println(io, "- Train dataset ID: `$(dataset_id(TRAIN_SPLIT))`")
    println(io, "\nValidation selects checkpoints. The final test split is evaluated only after the training configuration and checkpoint are fixed.")
  end
end

function dummy_dataset(split, region=nothing)
  item = if region === nothing
    example_at(split, 1)
  else
    idx = findfirst(==(region), split.regions)
    example_at(split, idx)
  end
  return Dict(item)
end

# ---------------------------------------------------------------------------
# Display abstraction: TUI.GPUBoard vs ProgressMeter bars. Each call site
# below dispatches on the board's type, so train_model/run_all_gpus stay
# backend-agnostic; only these functions know about TUI vs ProgressMeter.
# ---------------------------------------------------------------------------

function make_display(device_labels::Vector{String})
  DISPLAY_MODE == "tui" && return TUI.GPUBoard(device_labels)
  bars = [ProgressMeter.Progress(1; desc="$(dev) ", offset=i - 1, showspeed=true)
          for (i, dev) in enumerate(device_labels)]
  for i in eachindex(bars)
    ProgressMeter.update!(bars[i], 0; showvalues=[(:job, "idle")])
  end
  return bars
end

display_skip!(board::TUI.GPUBoard, slot, dev, scope_name, representation, maxiters) =
  TUI.update!(board, slot; variant="$(scope_name)-$(representation) already complete ✓",
              iter=maxiters, max_iter=maxiters)
function display_skip!(bars::Vector{ProgressMeter.Progress}, slot, dev, scope_name, representation, maxiters)
  bars[slot] = ProgressMeter.Progress(maxiters; desc="$(dev) ", offset=slot - 1, showspeed=true)
  ProgressMeter.update!(bars[slot], maxiters;
    showvalues=[(:job, "$(scope_name)-$(representation) already complete ✓"),
                (:epoch, "$(EPOCHS)/$(EPOCHS)")])
end

function display_start(board::TUI.GPUBoard, slot, dev, scope_name, representation, maxiters, batches)
  TUI.update!(board, slot; variant="$(scope_name)-$(representation)",
              iter=0, max_iter=maxiters, loss=Float32(NaN))
  return (state, loss) -> begin
    TUI.update!(board, slot; iter=Int(state.iter), max_iter=maxiters, loss=Float32(loss))
    false
  end
end
function display_start(bars::Vector{ProgressMeter.Progress}, slot, dev, scope_name, representation, maxiters, batches)
  bars[slot] = ProgressMeter.Progress(maxiters; desc="$(dev) ", offset=slot - 1, showspeed=true)
  return (state, loss) -> begin
    iter = Int(state.iter)
    epoch = iter ÷ batches
    ProgressMeter.update!(bars[slot], iter;
      showvalues=[(:job, "$(scope_name)-$(representation)"),
                  (:epoch, "$(epoch)/$(EPOCHS)"),
                  (:loss, round(Float32(loss); digits=6))])
    false
  end
end

display_finish!(board::TUI.GPUBoard, slot, scope_name, representation, maxiters) =
  TUI.update!(board, slot; variant="$(scope_name)-$(representation) ✓", iter=maxiters, max_iter=maxiters)
display_finish!(bars::Vector{ProgressMeter.Progress}, slot, scope_name, representation, maxiters) =
  ProgressMeter.finish!(bars[slot];
    showvalues=[(:job, "$(scope_name)-$(representation) ✓"), (:epoch, "$(EPOCHS)/$(EPOCHS)")])

display_fail!(board::TUI.GPUBoard, slot, label) = TUI.update!(board, slot; variant=label)
display_fail!(bars::Vector{ProgressMeter.Progress}, slot, label) =
  ProgressMeter.finish!(bars[slot]; showvalues=[(:job, label)])

function train_model(representation::Symbol, scope::Symbol;
                     region::Union{Symbol,Nothing}=nothing,
                     board::Union{TUI.GPUBoard,Vector{ProgressMeter.Progress},Nothing}=nothing,
                     device_labels::Union{Vector{String},Nothing}=nothing,
                     gpu_slot::Int=1)
  count = region === nothing ? TRAIN_SPLIT.size : family_size(TRAIN_SPLIT, region)
  batches = cld(count, BATCH_SIZE)
  maxiters = batches * EPOCHS
  parameter_count = representation === :power_series ? 9_877 : 8_772
  parameter_tag = "p$(lpad(parameter_count, 6, '0'))"
  dataset = dummy_dataset(TRAIN_SPLIT, region)
  settings = PINNSettings(
    NEURON_COUNT, MODEL_SEED, dataset, maxiters, N, N + 1, NUM_POINTS,
    0f0, 1f0, 1f0, 1f0, collect(range(0f0, 1f0, length=NUM_POINTS)),
    "adam", representation, :trace_determinant,
  )

  scope_name = region === nothing ? "global" : String(region)
  output_dir = region === nothing ?
    joinpath(OUTPUT_ROOT, "global", String(representation)) :
    joinpath(OUTPUT_ROOT, "family", String(region), String(representation))
  mkpath(output_dir)
  final_model_path = joinpath(output_dir, "model-dense-mlp-$(parameter_tag).checkpoint")
  if isfile(final_model_path)
    @info "Skipping completed model" representation scope=scope_name path=final_model_path
    if board !== nothing
      display_skip!(board, gpu_slot, device_labels[gpu_slot], scope_name, representation, maxiters)
    end
    return final_model_path
  end

  provider = region === nothing ?
    ((epoch, batch) -> batch_items(TRAIN_SPLIT, epoch, batch, BATCH_SIZE)) :
    ((epoch, batch) -> family_batch_items(TRAIN_SPLIT, region, epoch, batch, BATCH_SIZE))

  common_metadata(epoch, iteration) = Dict{String,Any}(
    "study" => "dual-representation",
    "scope" => String(scope),
    "family" => region === nothing ? "all" : String(region),
    "dataset_id" => dataset_id(TRAIN_SPLIT),
    "dataset_seed" => DATASET_SEED,
    "shuffle_seed" => SHUFFLE_SEED,
    "model_seed" => MODEL_SEED,
    "model_type" => "dense_mlp",
    "hidden_layers" => [64, 64, 64],
    "input_encoding" => "trace_determinant",
    "input_width" => 2,
    "output_width" => representation === :power_series ? N + 1 : 4,
    "parameter_count" => parameter_count,
    "checkpoint_filename_tag" => "dense-mlp-$(parameter_tag)",
    "unique_training_odes" => count,
    "batch_size" => BATCH_SIZE,
    "epoch" => epoch,
    "epochs_planned" => EPOCHS,
    "optimizer_updates" => iteration,
    "example_exposures" => (iteration ÷ batches) * count +
                           min((iteration % batches) * BATCH_SIZE, count),
    "power_series_degree" => N,
    "objective_components" => "pde + supervised",
    "diagnostic_components" => "bc",
  )

  function save_intermediate(p, iteration, net, _st, _run_id)
    epoch = iteration ÷ batches
    epoch == EPOCHS && return
    path = joinpath(output_dir, "snapshots",
                    "epoch-$(lpad(epoch, 4, '0'))-iter-$(lpad(iteration, 8, '0'))-dense-mlp-$(parameter_tag).checkpoint")
    PINN.SafeTensorSnapshots.save_checkpoint(path, p, net, MODEL_SEED;
      representation=representation, iteration=iteration,
      extra_metadata=common_metadata(epoch, iteration))
    @info "Saved dual-representation checkpoint" scope=scope_name representation epoch iteration path
  end

  interrupted = Ref(false)
  function save_interrupted(p, iteration, net, _st, _run_id)
    interrupted[] = true
    epoch = iteration ÷ batches
    path = joinpath(output_dir, "interrupted-epoch-$(lpad(epoch, 4, '0'))-iter-$(lpad(iteration, 8, '0'))-dense-mlp-$(parameter_tag).checkpoint")
    PINN.SafeTensorSnapshots.save_checkpoint(path, p, net, MODEL_SEED;
      representation=representation, iteration=iteration,
      extra_metadata=common_metadata(epoch, iteration))
  end

  @info "Starting matched model" representation scope=scope_name count batches epochs=EPOCHS maxiters
  progress_callback = board === nothing ? nothing :
    display_start(board, gpu_slot, device_labels[gpu_slot], scope_name, representation, maxiters, batches)
  p, net, st, _, _ = train_pinn(
    settings, output_dir;
    run_id="dual-$(scope_name)-$(representation)",
    on_milestone=save_intermediate,
    on_interrupt=save_interrupted,
    batch_size=BATCH_SIZE,
    snapshot_epoch_interval=CHECKPOINT_EPOCH_INTERVAL,
    batch_provider=provider,
    streaming_dataset_size=count,
    write_loss_csv=true,
    progress_callback=progress_callback,
  )
  if !interrupted[]
    PINN.SafeTensorSnapshots.save_checkpoint(
      final_model_path, p, net, MODEL_SEED;
      representation=representation, iteration=maxiters,
      extra_metadata=common_metadata(EPOCHS, maxiters),
    )
    if board !== nothing
      display_finish!(board, gpu_slot, scope_name, representation, maxiters)
    end
  end
end

write_run_manifests()

function build_jobs()
  jobs = NamedTuple[]
  if RUN_GLOBAL_MODELS
    for representation in (:power_series, :eigenvalue)
      push!(jobs, (representation=representation, scope=:global, region=nothing))
    end
  end
  if RUN_FAMILY_MODELS
    for region in REGIONS
      for representation in (:power_series, :eigenvalue)
        push!(jobs, (representation=representation, scope=:family, region=region))
      end
    end
  end
  return jobs
end

function run_all_gpus(jobs)
  CUDA.functional() || error("The 10M run requires CUDA, but CUDA.functional() is false")
  devices = collect(CUDA.devices())
  available = length(devices)
  available >= GPU_COUNT || error("The run requires $(GPU_COUNT) GPUs, but CUDA.jl sees only $(available)")
  Threads.nthreads() >= GPU_COUNT || error(
    "The run requires at least $(GPU_COUNT) Julia threads; launch with `julia --project=. -t $(GPU_COUNT) scripts/shared/run_all.jl`"
  )

  selected_devices = devices[1:GPU_COUNT]
  # Fail fast before starting any long-running model. This exercises the
  # eigenvalue loss's GPU forward and reverse passes, where unsupported generic
  # adjoints would otherwise fail only after the multi-GPU queue has started.
  CUDA.device!(selected_devices[1])
  smoke_items = batch_items(TRAIN_SPLIT, 1, 1, 4)
  smoke_settings = PINNSettings(
    4, MODEL_SEED, Dict(smoke_items[1]), 1, N, N + 1, 3,
    0f0, 1f0, 1f0, 1f0, Float32[0f0, 0.5f0, 1f0],
    "adam", :eigenvalue, :trace_determinant,
  )
  smoke_buf = PINN.loss_functions.precompute_batch_eig_buffers(
    smoke_settings, smoke_items, true, CUDA.cu)
  # Include negative k explicitly: spirals and centers require k < 0. A
  # positive-only random smoke test cannot expose NaNs in the power pullback.
  smoke_out = CUDA.cu(Float32[
    -1.0 -0.5  0.5  1.0
    -2.0 -0.5  0.5  3.0
     1.0  1.0  1.0  1.0
     1.0 -1.0  0.5 -0.5
  ])
  smoke_objective(o) = sum(PINN.loss_functions.batched_eigenvalue_losses(o, smoke_buf))
  smoke_loss, smoke_gradient = Zygote.withgradient(smoke_objective, smoke_out)
  smoke_grad = smoke_gradient[1]
  CUDA.synchronize()
  isfinite(smoke_loss) || error("Eigenvalue GPU gradient preflight produced a non-finite loss")
  smoke_grad_cpu = Array(smoke_grad)
  all(isfinite, smoke_grad_cpu) || error(
    "Eigenvalue GPU gradient preflight produced non-finite values: $(smoke_grad_cpu)")
  smoke_buf = smoke_out = smoke_grad = nothing
  GC.gc(true)
  CUDA.reclaim()
  @info "Eigenvalue GPU gradient preflight passed"

  device_labels = ["GPU $(i - 1) ($(CUDA.name(selected_devices[i])))" for i in 1:GPU_COUNT]
  @info "Live display" mode=DISPLAY_MODE
  board = make_display(device_labels)
  queue = Channel{NamedTuple}(length(jobs))
  foreach(job -> put!(queue, job), jobs)
  close(queue)

  failures = Channel{Any}(length(jobs))
  workers = Task[]
  for slot in 1:GPU_COUNT
    push!(workers, Threads.@spawn begin
      CUDA.device!(selected_devices[slot])
      for job in queue
        try
          train_model(job.representation, job.scope;
                      region=job.region, board=board, device_labels=device_labels, gpu_slot=slot)
        catch err
          display_fail!(board, slot,
            "$(job.region === nothing ? "global" : job.region)-$(job.representation) FAILED")
          bt = catch_backtrace()
          @error "Dual-representation model failed" gpu=slot job exception=(err, bt)
          put!(failures, (job=job, error=err, backtrace=bt))
        finally
          # All persisted checkpoints contain CPU weights. Finish outstanding
          # kernels, collect model/batch objects, then return cached device
          # allocations to CUDA before this worker accepts its next model.
          try
            CUDA.synchronize()
          catch
          end
          GC.gc(true)
          try
            CUDA.reclaim()
          catch reclaim_error
            @warn "Could not fully reclaim CUDA memory after model job" gpu=slot exception=reclaim_error
          end
        end
      end
    end)
  end
  foreach(fetch, workers)
  close(failures)

  failed = collect(failures)
  isempty(failed) || error("$(length(failed)) model job(s) failed; see errors above")
end

run_all_gpus(build_jobs())

@info "Dual-representation pretraining complete" output_root=OUTPUT_ROOT
