# =============================================================================
# utils/variants.jl — Variant registry + per-GPU dispatch
#
# A variant is a leaf in a tree:
#
#     representation (:power_series | :eigenvalue)   ← root, fixes loss form
#       neuron_count, N, maxiters, seed_offset      ← free parameters
#
# The variant name (e.g. "ps_N20", "eig") is the durable identifier that
# goes into checkpoint metadata and output directory names.
#
# `build_jobs(specs, dataset, ...)` resolves a list of VariantSpecs into
# VariantJobs with a round-robin device assignment. `run_staged_variants`
# spawns one worker task per job, binds CUDA.device! per the assignment, and
# drives a single TUI.GPUBoard so all devices are visible simultaneously.
#
# Each worker task calls CUDA.device!(i) FIRST, before touching any other CUDA
# state — see plans/active/staged-gpu-runs.md II.1 (task-local device + the
# default-device hazard). On CPU-only boxes CUDA.functional() is false; the
# device_names() fallback returns a single "CPU" slot and workers stay
# sequential (one CPU "GPU" row, jobs dispatched in order).
# =============================================================================

module Variants

include("../architectures/PINN.jl")
using .PINN

include("../utils/tui.jl")
using .TUI

include("../utils/helper_funcs.jl")
using .helper_funcs

include("../utils/plugboard.jl")
using .Plugboard

using CUDA
using JSON
using Dates
using Random

export VariantSpec, VariantJob,
       build_jobs, run_staged_variants,
       device_names, generate_shared_dataset, write_manifest

# ---------------------------------------------------------------------------
# VariantSpec — declarative leaf of the variant tree
# ---------------------------------------------------------------------------

"""
    VariantSpec(name, representation; N=20, neuron_count=64, maxiters=3000, seed_offset=0)

One variant to train. `name` is the durable identifier (used for the output
directory and as the variant slot label in the GPUBoard). `representation`
fixes the encode/loss/reconstruct triple — see plans/active/staged-gpu-runs.md
III.1 — so all four inseparable operations move together.
"""
struct VariantSpec
  name::String
  representation::Symbol
  N::Int
  neuron_count::Int
  maxiters::Int
  seed_offset::Int
end

function VariantSpec(name::AbstractString, representation::Symbol;
                     N::Int=20, neuron_count::Int=64, maxiters::Int=3000,
                     seed_offset::Int=0)
  representation in (:power_series, :eigenvalue) || error(
    "VariantSpec: representation must be :power_series or :eigenvalue, got :$representation")
  return VariantSpec(String(name), representation, N, neuron_count, maxiters, seed_offset)
end

# ---------------------------------------------------------------------------
# VariantJob — resolved, ready-to-run
# ---------------------------------------------------------------------------

"""
    VariantJob

Resolved variant: `spec` + a `device_id` (1-indexed, into the visible device
space returned by `device_names()`) + an `output_dir` + a shared `dataset`.
The dataset is held by reference — all variants train on the SAME ODE pool.
"""
struct VariantJob
  tag::String
  spec::VariantSpec
  device_id::Int
  output_dir::String
  dataset::Dict
end

# ---------------------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------------------

"""
    device_names() → Vector{String}

Human-readable device names, one per visible CUDA device
(`CUDA_VISIBLE_DEVICES` renumbers physical → visible; we use the visible
space, which is what CUDA.device! accepts). Falls back to ["CPU"] when no
CUDA is present.
"""
function device_names()
  if !CUDA.functional()
    return String["CPU"]
  end
  n = CUDA.ndevices()
  return [string("GPU ", i, " (", CUDA.name(CUDA.device(i - 1)), ")") for i in 0:(n - 1)]
end

# ---------------------------------------------------------------------------
# Shared dataset generation
# ---------------------------------------------------------------------------

"""
    generate_shared_dataset(; n_per_region::Int=200, num_terms::Int=21,
                            regions=TRACE_DET_REGIONS, seed::Int=1234) → Dict

Generate a single fixed, seeded ODE dataset, shared by every variant in a
run. Sampling across `regions` (default: all six trace-determinant regions)
gives a balanced pool; `num_terms+1` matches the supervised target length.

Deterministic given `seed`: same seed → same dataset, so runs are
reproducible and a re-run can pick up where it left off.
"""
function generate_shared_dataset(; n_per_region::Int=200, num_terms::Int=21,
                                 regions=Plugboard.TRACE_DET_REGIONS,
                                 seed::Int=1234)
  rng = MersenneTwister(seed)
  return Plugboard.generate_region_dataset(regions, n_per_region, num_terms;
                                          a0=1.0f0, a1=0.0f0,
                                          tau_lim=2.0f0, delta_lim=2.0f0,
                                          rng=rng)
end

# ---------------------------------------------------------------------------
# Job-list builder (round-robin device assignment)
# ---------------------------------------------------------------------------

"""
    build_jobs(specs, dataset, output_root; n_devices=0) → Vector{VariantJob}

Resolve a list of `VariantSpec`s into `VariantJob`s with round-robin device
assignment over the visible CUDA devices (or one CPU slot when no GPU).

Round-robin gives a stable assignment (the same spec list always lands on
the same slots) and matches the "one model per GPU" pattern (Q2 in the
staged-gpu-runs plan). When jobs outnumber devices, later jobs queue on
the same devices — the GPUBoard then shows the LAST-loaded variant on each
device, which is what users expect from the cashier-board metaphor.
"""
function build_jobs(specs::AbstractVector{VariantSpec}, dataset::Dict,
                    output_root::AbstractString;
                    n_devices::Int=0)
  if n_devices <= 0
    n_devices = length(device_names())
  end
  jobs = VariantJob[]
  for (i, spec) in enumerate(specs)
    dev = ((i - 1) % n_devices) + 1      # 1-indexed into device_names()
    dir = joinpath(output_root, "staged-", spec.name)
    push!(jobs, VariantJob(spec.name, spec, dev, dir, dataset))
  end
  return jobs
end

# ---------------------------------------------------------------------------
# run_staged_variants — per-GPU dispatch + GPUBoard render
# ---------------------------------------------------------------------------

"""
    run_staged_variants(jobs::Vector{VariantJob}; benchmark_dataset=nothing)

Spawn one worker task per job. Each worker:

1. Binds `CUDA.device!(job.device_id - 1)` as its FIRST CUDA action (per
   staged-gpu-runs.md II.1 rule 1; on CPU-only boxes the binding is a no-op).
2. Builds its `PINNSettings` from the job spec + shared dataset.
3. Calls `train_pinn` with a `progress_callback` that updates the GPUBoard
   slot's iter + loss.
4. Saves a checkpoint under `job.output_dir/model.checkpoint`.

The board is created once with one row per visible device; updates are
serialized by the board's lock. The function blocks until all jobs finish.
"""
function run_staged_variants(jobs::AbstractVector{VariantJob};
                             benchmark_dataset::Union{Dict,Nothing}=nothing)
  devs = device_names()
  max_iter = isempty(jobs) ? 10000 : jobs[1].spec.maxiters
  board = TUI.GPUBoard(devs; max_iter=max_iter)

  tasks = Task[]
  for (slot, job) in enumerate(jobs)
    push!(tasks, Threads.@spawn begin
      try
        if CUDA.functional()
          CUDA.device!(job.device_id - 1)        # 0-indexed for CUDA.jl
        end
        _run_one_job!(board, job, benchmark_dataset)
      catch e
        @error "Variant $(job.tag) failed on device $(job.device_id)" exception=(e, catch_backtrace())
        TUI.update!(board, job.device_id; variant=string(job.tag, " FAILED"))
      end
    end)
  end
  foreach(fetch, tasks)

  return jobs
end

# Worker body — kept private; called inside a task spawned by run_staged_variants.
function _run_one_job!(board::TUI.GPUBoard, job::VariantJob,
                       benchmark_dataset::Union{Dict,Nothing})
  mkpath(job.output_dir)
  slot = job.device_id      # GPUBoard rows are indexed by device, not by job
  TUI.update!(board, slot;
              variant=job.tag,
              max_iter=job.spec.maxiters,
              iter=0, loss=Float32(NaN))

  settings = _build_settings(job)

  # Progress callback: forward iter + loss to the board.
  # The Optimization.jl callback receives (state, l); `state.iter` is the current
  # iteration index.
  progress_cb = function(state, l)
    iter = Int(getfield(state, :iter))
    TUI.update!(board, slot; iter=iter, loss=Float32(l))
    return false
  end

  # Train — checkpointing off; per-job CSV is enough for staged runs.
  p_trained, coeff_net, st, _run_id, _history = train_pinn(
    settings, job.output_dir;
    run_id=job.tag,
    write_loss_csv=true,
    snapshot_epoch_interval=0,
    progress_callback=progress_cb,
  )

  # Save the final model. Representations come through metadata so the
  # checkpoint loader can reconstruct u(x) for either rep.
  PINN.SafeTensorSnapshots.save_checkpoint(
    joinpath(job.output_dir, "model.checkpoint"),
    p_trained, coeff_net, settings.seed;
    representation=settings.representation,
    extra_metadata=Dict{String,Any}(
      "objective_components" => "pde + supervised",
      "diagnostic_components" => "bc",
      "variant_name" => job.tag,
    )
  )

  # Final slot update: mark complete + final loss.
  final_loss = isempty(_history) ? Float32(NaN) : Float32(_history[end].total)
  TUI.update!(board, slot;
              variant=string(job.tag, " ✓"),
              iter=job.spec.maxiters,
              loss=final_loss)
end

function _build_settings(job::VariantJob)
  xs = collect(range(Float32(0.0), Float32(1.0), length=22))
  return PINNSettings(
    job.spec.neuron_count,
    1234 + job.spec.seed_offset,
    Dict{Any,Any}(job.dataset),
    job.spec.maxiters,
    job.spec.N,
    job.spec.N + 1,
    22,                                 # num_points (≥ N+1)
    Float32(0.0), Float32(1.0),
    Float32(1.0), Float32(1.0),         # supervised / pde weights
    xs, "adam", job.spec.representation,
  )
end

# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------

"""
    write_manifest(path, jobs, dataset_meta; n_devices)

Persist a run manifest: job list, device assignments, dataset metadata.
Lets a future restart decide which jobs are done / running / fresh.
"""
function write_manifest(path::AbstractString, jobs::AbstractVector{VariantJob},
                        dataset_meta::Dict; n_devices::Int=length(device_names()))
  entries = [
    Dict(
      "tag" => j.tag,
      "representation" => String(j.spec.representation),
      "N" => j.spec.N,
      "neuron_count" => j.spec.neuron_count,
      "maxiters" => j.spec.maxiters,
      "seed_offset" => j.spec.seed_offset,
      "device_id" => j.device_id,
      "output_dir" => j.output_dir,
    ) for j in jobs
  ]
  payload = Dict(
    "created_at" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
    "n_devices" => n_devices,
    "device_names" => device_names(),
    "dataset" => dataset_meta,
    "jobs" => entries,
  )
  open(path, "w") do io
    JSON.print(io, payload, 2)
  end
  return path
end

end # module Variants