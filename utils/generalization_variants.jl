# =============================================================================
# utils/generalization_variants.jl — Generalization experiments × representations
#
# The five diagnostic shapes from `utils/experiments.jl` (transfer, extrapolate,
# cross-region-range, showcase, gen_radius) each test a different facet of
# generalization. They were originally run sequentially per rep
# (`scripts/shared/run_all.jl`); this module threads them through the same
# per-GPU variant runner that staged_variants.jl uses for single-ODE training:
#
#     rep × experiment   →   GeneralizationSpec
#                            ↓
#                          GenJob (spec + device_id + output_dir)
#                            ↓
#                          run_generalization_variants (one task per job)
#                            ↓
#                          GPUBoard (one slot per visible device)
#
# Each worker task binds CUDA.device!(i) FIRST (per staged-gpu-runs.md II.1
# rule 1) and then dispatches on spec.name to the matching run_* function.
# A new instance of Experiments is built per rep — the two representations
# share NO mutable state, so they can run truly concurrently.
# =============================================================================

module GeneralizationVariants

include("../utils/tui.jl")
using .TUI

include("../utils/experiments.jl")
using .Experiments

include("../utils/plugboard.jl")
using .Plugboard

using CUDA
using JSON
using Dates

export GeneralizationSpec, GenJob,
       build_gen_jobs, run_generalization_variants,
       default_generalization_specs, REGIONS

const REGIONS = Plugboard.TRACE_DET_REGIONS

# ---------------------------------------------------------------------------
# Spec — declarative (experiment, rep, params) tuple
# ---------------------------------------------------------------------------

"""
    GeneralizationSpec(name, representation; regions=REGIONS, Rmax=6, Ng=31,
                       n_per_region=200, maxiters=3000)

One generalization experiment for one representation. `name` is the diagnostic
shape: `:transfer`, `:extrapolate`, `:range` (cross-region-range),
`:showcase`, or `:gen_radius`.

Fields:
- `name::Symbol`            — which experiment function to dispatch to
- `representation::Symbol`  — `:power_series` or `:eigenvalue`
- `regions::Vector{Symbol}` — train/test regions (used by transfer/extrap/range)
- `Rmax::Int`               — number of extrapolation shells (extrap/range)
- `Ng::Int`                 — grid size for gen_radius
- `n_per_region::Int`       — ODEs per region (forwarded to ExperimentConfig)
- `maxiters::Int`           — Adam iterations per PINN (forwarded to ExperimentConfig)

Unused fields default silently — `:showcase` ignores `regions`/`Rmax`/`Ng`
entirely, while `:transfer` ignores `Rmax`/`Ng`.
"""
struct GeneralizationSpec
  name::Symbol
  representation::Symbol
  regions::Vector{Symbol}
  Rmax::Int
  Ng::Int
  n_per_region::Int
  maxiters::Int
end

function GeneralizationSpec(name::Symbol, representation::Symbol;
                            regions::Vector{Symbol}=REGIONS,
                            Rmax::Int=6, Ng::Int=31,
                            n_per_region::Int=200, maxiters::Int=3000)
  name in (:transfer, :extrapolate, :range, :showcase, :gen_radius) || error(
    "GeneralizationSpec: unknown experiment :$name; expected one of " *
    ":transfer, :extrapolate, :range, :showcase, :gen_radius")
  representation in (:power_series, :eigenvalue) || error(
    "GeneralizationSpec: representation must be :power_series or :eigenvalue, got :$representation")
  return GeneralizationSpec(name, representation, copy(regions), Rmax, Ng,
                            n_per_region, maxiters)
end

# ---------------------------------------------------------------------------
# Default spec list — both reps × all 5 experiments
# ---------------------------------------------------------------------------

"""
    default_generalization_specs(; reps=(:power_series, :eigenvalue),
                                   experiments=(:transfer, :extrapolate,
                                                :range, :showcase, :gen_radius),
                                   kwargs...)

Cross-product of reps × experiments. The user can override either axis to
narrow the run; `kwargs...` are forwarded to `GeneralizationSpec` for the
fields that apply (e.g. `Rmax=4` to use a smaller shell count).
"""
function default_generalization_specs(;
                                      reps::Tuple=(:power_series, :eigenvalue),
                                      experiments::Tuple=(:transfer, :extrapolate,
                                                          :range, :showcase, :gen_radius),
                                      kwargs...)
  specs = GeneralizationSpec[]
  for rep in reps, exp in experiments
    push!(specs, GeneralizationSpec(exp, rep; kwargs...))
  end
  return specs
end

# ---------------------------------------------------------------------------
# GenJob — resolved, ready-to-run
# ---------------------------------------------------------------------------

"""
    GenJob

Resolved generalization job: `spec` + a `device_id` (1-indexed into the
visible device space) + an `output_dir`.
"""
struct GenJob
  spec::GeneralizationSpec
  device_id::Int
  output_dir::String
end

# ---------------------------------------------------------------------------
# Job-list builder (round-robin device assignment, matches Variants.jl)
# ---------------------------------------------------------------------------

"""
    build_gen_jobs(specs, output_root; n_devices=0) → Vector{GenJob}

Resolve a list of `GeneralizationSpec`s into `GenJob`s with round-robin device
assignment over the visible CUDA devices. Identical convention to
`Variants.build_jobs` — the same set of specs lands on the same slots across
runs.
"""
function build_gen_jobs(specs::AbstractVector{GeneralizationSpec},
                        output_root::AbstractString;
                        n_devices::Int=0)
  if n_devices <= 0
    n_devices = Variants_device_count()
  end
  jobs = GenJob[]
  for (i, spec) in enumerate(specs)
    dev = ((i - 1) % n_devices) + 1
    # Use string concat (not joinpath) so "-" stays a separator, not a literal
    # directory name. joinpath("/tmp", "x", "-", "y") → "/tmp/x/-/y".
    dir = joinpath(output_root, "gen-" * String(spec.name) * "-" * String(spec.representation))
    push!(jobs, GenJob(spec, dev, dir))
  end
  return jobs
end

# Local helper to avoid pulling Variants into this module's namespace.
function Variants_device_count()
  if CUDA.functional()
    return CUDA.ndevices()
  else
    return 1
  end
end

# ---------------------------------------------------------------------------
# run_generalization_variants — per-GPU dispatch
# ---------------------------------------------------------------------------

"""
    run_generalization_variants(jobs::Vector{GenJob})

Spawn one worker task per job. Each worker:

1. Binds `CUDA.device!(job.device_id - 1)` as its FIRST CUDA action
   (per staged-gpu-runs.md II.1 rule 1).
2. Builds an `ExperimentConfig` for the spec's representation.
3. Dispatches on `spec.name` to the matching `Experiments.run_*` function.
4. Saves the resulting PanelSet to `job.output_dir/panelset.json`.
5. Updates the GPUBoard slot for its device.

The board is created once with one row per visible device; updates are
serialized by the board's lock. The function blocks until all jobs finish.
"""
function run_generalization_variants(jobs::AbstractVector{GenJob})
  devs = String[]
  if CUDA.functional()
    n = CUDA.ndevices()
    devs = [string("GPU ", i, " (", CUDA.name(CUDA.device(i - 1)), ")") for i in 0:(n - 1)]
  else
    push!(devs, "CPU")
  end

  board = TUI.GPUBoard(devs)

  tasks = Task[]
  for job in jobs
    push!(tasks, Threads.@spawn begin
      try
        if CUDA.functional()
          CUDA.device!(job.device_id - 1)
        end
        _run_one_gen_job!(board, job)
      catch e
        @error "Generalization $(job.spec.name)/$(job.spec.representation) failed on device $(job.device_id)" exception=(e, catch_backtrace())
        TUI.update!(board, job.device_id;
                    variant=string(job.spec.name, "_", job.spec.representation, " FAILED"))
      end
    end)
  end
  foreach(fetch, tasks)

  return jobs
end

function _run_one_gen_job!(board::TUI.GPUBoard, job::GenJob)
  mkpath(job.output_dir)
  cfg = Experiments.ExperimentConfig(;
    representation=job.spec.representation,
    n_per_region=job.spec.n_per_region,
    maxiters=job.spec.maxiters,
  )
  tag = string(job.spec.name, "_", job.spec.representation)
  TUI.update!(board, job.device_id; variant=tag)

  ps = if job.spec.name === :transfer
    Experiments.run_transfer(cfg, job.spec.regions, job.spec.regions, tag;
                             output_root=job.output_dir)
  elseif job.spec.name === :extrapolate
    Experiments.run_extrapolate(cfg, job.spec.regions, job.spec.Rmax, tag;
                                output_root=job.output_dir)
  elseif job.spec.name === :range
    Experiments.run_cross_region_range(cfg, job.spec.regions, job.spec.regions,
                                       job.spec.Rmax, tag;
                                       output_root=job.output_dir)
  elseif job.spec.name === :showcase
    Experiments.run_showcase(cfg, tag; output_root=job.output_dir)
  elseif job.spec.name === :gen_radius
    Experiments.run_gen_radius(cfg, tag;
                               output_root=job.output_dir, Ng=job.spec.Ng)
  else
    error("Internal: spec.name :$(job.spec.name) should have been validated at construction")
  end

  # Save the PanelSet next to the experiment's training artifacts so the
  # viewer can load it the same way it loads data/shared_*.json.
  out_path = joinpath(job.output_dir, "panelset.json")
  Experiments.save_experiment(out_path, ps)

  TUI.update!(board, job.device_id; variant=string(tag, " ✓"))
  return ps
end

end # module GeneralizationVariants