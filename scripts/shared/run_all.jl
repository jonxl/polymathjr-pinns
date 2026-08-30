# Large, matched dual-representation pretraining run.
#
# This script trains reusable frozen models. Generalization experiments load
# these checkpoints later and perform inference; they do not retrain them.

using Dates
using JSON

include("../../architectures/PINN.jl")
using .PINN
include("../../utils/dual_dataset.jl")
using .DualDataset

const REGIONS = [:saddle, :stable_node, :unstable_node,
                 :stable_spiral, :unstable_spiral, :center]

const OUTPUT_ROOT = "results/dual-representation-10m"
const TRAIN_SIZE = 10_000_000
const VALIDATION_SIZE = 500_000
const TEST_SIZE = 1_000_000
const BATCH_SIZE = 8_192
const EPOCHS = 100
const CHECKPOINT_EPOCH_INTERVAL = 5
const N = 20
const NEURON_COUNT = 64
const NUM_POINTS = 50
const TRAIN_SEED = 10_001
const VALIDATION_SEED = 20_001
const TEST_SEED = 30_001
const MODEL_SEED = 1_234
const RUN_GLOBAL_MODELS = true
const RUN_FAMILY_MODELS = true

const TRAIN_SPLIT = CanonicalODESplit(:train, TRAIN_SIZE, TRAIN_SEED, N; regions=REGIONS)
const VALIDATION_SPLIT = CanonicalODESplit(:validation, VALIDATION_SIZE, VALIDATION_SEED, N; regions=REGIONS)
const TEST_SPLIT = CanonicalODESplit(:test, TEST_SIZE, TEST_SEED, N; regions=REGIONS)

split_manifest(split) = Dict(
  "name" => String(split.name),
  "size" => split.size,
  "seed" => string(split.seed),
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

function train_model(representation::Symbol, scope::Symbol; region::Union{Symbol,Nothing}=nothing)
  count = region === nothing ? TRAIN_SPLIT.size : family_size(TRAIN_SPLIT, region)
  batches = cld(count, BATCH_SIZE)
  maxiters = batches * EPOCHS
  dataset = dummy_dataset(TRAIN_SPLIT, region)
  settings = PINNSettings(
    NEURON_COUNT, MODEL_SEED, dataset, maxiters, N, N + 1, NUM_POINTS,
    0f0, 1f0, 1f0, 1f0, collect(range(0f0, 1f0, length=NUM_POINTS)),
    "adam", representation,
  )

  scope_name = region === nothing ? "global" : String(region)
  output_dir = region === nothing ?
    joinpath(OUTPUT_ROOT, "global", String(representation)) :
    joinpath(OUTPUT_ROOT, "family", String(region), String(representation))
  mkpath(output_dir)

  provider = region === nothing ?
    ((epoch, batch) -> batch_items(TRAIN_SPLIT, epoch, batch, BATCH_SIZE)) :
    ((epoch, batch) -> family_batch_items(TRAIN_SPLIT, region, epoch, batch, BATCH_SIZE))

  common_metadata(epoch, iteration) = Dict{String,Any}(
    "study" => "dual-representation",
    "scope" => String(scope),
    "family" => region === nothing ? "all" : String(region),
    "dataset_id" => dataset_id(TRAIN_SPLIT),
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
                    "epoch-$(lpad(epoch, 4, '0'))-iter-$(lpad(iteration, 8, '0')).checkpoint")
    PINN.SafeTensorSnapshots.save_checkpoint(path, p, net, MODEL_SEED;
      representation=representation, iteration=iteration,
      extra_metadata=common_metadata(epoch, iteration))
    @info "Saved dual-representation checkpoint" scope=scope_name representation epoch iteration path
  end

  interrupted = Ref(false)
  function save_interrupted(p, iteration, net, _st, _run_id)
    interrupted[] = true
    epoch = iteration ÷ batches
    path = joinpath(output_dir, "interrupted-epoch-$(lpad(epoch, 4, '0'))-iter-$(lpad(iteration, 8, '0')).checkpoint")
    PINN.SafeTensorSnapshots.save_checkpoint(path, p, net, MODEL_SEED;
      representation=representation, iteration=iteration,
      extra_metadata=common_metadata(epoch, iteration))
  end

  @info "Starting matched model" representation scope=scope_name count batches epochs=EPOCHS maxiters
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
  )
  if !interrupted[]
    PINN.SafeTensorSnapshots.save_checkpoint(
      joinpath(output_dir, "model.checkpoint"), p, net, MODEL_SEED;
      representation=representation, iteration=maxiters,
      extra_metadata=common_metadata(EPOCHS, maxiters),
    )
  end
end

write_run_manifests()

for representation in (:power_series, :eigenvalue)
  RUN_GLOBAL_MODELS && train_model(representation, :global)
  if RUN_FAMILY_MODELS
    for region in REGIONS
      train_model(representation, :family; region=region)
    end
  end
end

@info "Dual-representation pretraining complete" output_root=OUTPUT_ROOT
