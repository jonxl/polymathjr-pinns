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
# Shared module definitions are loaded once; each job owns its parameters and buffers.
# =============================================================================

module GeneralizationVariants

if isdefined(parentmodule(@__MODULE__), :TUI)
  import ..TUI
else
  include("../utils/tui.jl")
end
using .TUI

if isdefined(parentmodule(@__MODULE__), :GeometryExperiments)
  import ..GeometryExperiments
else
  include("geometry_experiments.jl")
end
using .GeometryExperiments
const Experiments = GeometryExperiments.Experiments
using .Experiments
const GEOMETRY_EXPERIMENTS = GeometryExperiments.GeneralizationGeometry.EXPERIMENTS
const ALL_EXPERIMENTS = (:transfer, :extrapolate, :range, :showcase, :gen_radius, :gen_radius_family, GEOMETRY_EXPERIMENTS...)

if isdefined(parentmodule(@__MODULE__), :Plugboard)
  import ..Plugboard
else
  include("../utils/plugboard.jl")
end
using .Plugboard

if isdefined(parentmodule(@__MODULE__), :RunManagement)
  import ..RunManagement
else
  include("run_management.jl")
end
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
- `group::String`           — optional folder that sets this job apart from another with the
                              same experiment and variant, e.g. `"degree-25"` in a degree sweep

Unused fields default silently — `:showcase` ignores `regions`/`Rmax`/`Ng`
entirely, while `:transfer` ignores `Rmax`/`Ng`.
"""
struct GeneralizationSpec
  name::Symbol
  representation::Symbol
  variant::Symbol
  coefficient_weighting::Symbol
  regions::Vector{Symbol}
  Rmax::Int
  Ng::Int
  n_per_region::Int
  maxiters::Int
  config::Union{Nothing,Dict{String,Any}}
  group::String
end

function GeneralizationSpec(name::Symbol, representation::Symbol;
                            regions::Vector{Symbol}=REGIONS,
                            Rmax::Int=6, Ng::Union{Int,Nothing}=nothing,
                            n_per_region::Union{Int,Nothing}=nothing, maxiters::Int=3000, config=nothing,
                            group::AbstractString="")
  name in ALL_EXPERIMENTS || error(
    "GeneralizationSpec: unknown experiment :$name; expected one of " *
    ":transfer, :extrapolate, :range, :showcase, :gen_radius, $(GEOMETRY_EXPERIMENTS)")
  representation in (:power_series, :eigenvalue, :power_series_variation) || error(
    "GeneralizationSpec: representation must be :power_series, :eigenvalue or :power_series_variation, got :$representation")
  controls = name in GEOMETRY_EXPERIMENTS ? GeometryExperiments.GeneralizationGeometry.jeet_controls(name) : (ng=81,n=name === :gen_radius_family ? 100 : 200)
  Ng=something(Ng,controls.ng); n_per_region=something(n_per_region,controls.n)
  Ng >= 2 && n_per_region > 0 && maxiters > 0 || error("Invalid grid size, dataset size, or iteration limit")
  name === :showcase && n_per_region < 3 && error("showcase requires at least three examples (one per training region)")
  occursin(r"^[A-Za-z0-9_-]*$", group) || error("GeneralizationSpec: group must be a plain folder name, got \"$group\"")
  return GeneralizationSpec(name, Experiments.PINN.canonical_representation(representation), representation, Experiments.PINN.coefficient_weighting(representation), copy(regions), Rmax, Ng,
                            n_per_region, maxiters, config, String(group))
end

# ---------------------------------------------------------------------------
# Default spec list — both reps × all 5 experiments
# ---------------------------------------------------------------------------

"""
    default_generalization_specs(; reps=(:power_series, :eigenvalue, :power_series_variation),
                                   experiments=(:transfer, :extrapolate,
                                                :range, :showcase, :gen_radius),
                                   kwargs...)

Cross-product of reps × experiments. The user can override either axis to
narrow the run; `kwargs...` are forwarded to `GeneralizationSpec` for the
fields that apply (e.g. `Rmax=4` to use a smaller shell count).
"""
function default_generalization_specs(;
                                      reps::Tuple=(:power_series, :eigenvalue, :power_series_variation),
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
    dir = joinpath(output_root, spec.group, "gen-" * String(spec.name) * "-" * String(spec.variant))
    push!(jobs, GenJob(spec, dev, dir))
  end
  return jobs
end

"Manifest id of a job: its folder name, prefixed by the spec's group when it has one."
gen_job_id(job::GenJob) = isempty(job.spec.group) ? basename(job.output_dir) :
                          string(job.spec.group, "/", basename(job.output_dir))

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

`lanes` is how many jobs run at once on each device (default 1). The board is
created once with one row per device lane; updates are serialized by the
board's lock. The function blocks until all jobs finish.
"""
function run_generalization_variants(jobs::AbstractVector{GenJob}; run=nothing, lanes::Int=1)
  lanes >= 1 || error("run_generalization_variants: lanes must be at least 1, got $lanes")
  devs = String[]
  if CUDA.functional()
    n = CUDA.ndevices()
    devs = [string("GPU ", i, " (", CUDA.name(CUDA.CuDevice(i)), ")") for i in 0:(n - 1)]
  else
    push!(devs, "CPU")
  end

  rows = lanes == 1 ? devs : [string(dev, " #", lane) for dev in devs for lane in 1:lanes]
  board = TUI.GPUBoard(rows)

  # Each running job borrows one board row of its device and returns it when done.
  free_lanes = [Channel{Int}(lanes) for _ in devs]
  for pool in free_lanes, lane in 1:lanes
    put!(pool, lane)
  end

  return RunManagement.run_jobs(jobs; run=run, devices=length(devs), lanes=lanes, id=gen_job_id) do job
    CUDA.functional() && CUDA.device!(job.device_id-1)
    lane = take!(free_lanes[job.device_id])
    try
      _run_one_gen_job!(board, job; slot=(job.device_id - 1) * lanes + lane)
    finally
      put!(free_lanes[job.device_id], lane)
    end
  end
end

function config_keywords(spec)
  spec.config === nothing && return (representation=spec.variant,n_per_region=spec.n_per_region,maxiters=spec.maxiters)
  c=spec.config; m=c["model"]; o=c["objective"]; d=c["data"]; t=c["training"]
  P=Experiments.PINN
  return (representation=Symbol(m["representation"]), N=Int(m["degree"]), neuron_count=Int(first(m["hidden_widths"])),
    model=P.ModelConfig(hidden_widths=m["hidden_widths"],activation=Symbol(m["activation"])),
    loss=P.LossConfig(coefficient_weighting=Symbol(o["coefficient_weighting"]), ic_weight=o["ic_weight"]),
    supervised_weight=Float32(o["supervised_weight"]),pde_weight=Float32(o["pde_weight"]),
    seed=Int(m["seed"]),data_seed=Int(d["seed"]),num_points=Int(d["num_points"]),x_left=Float32(d["x_left"]),x_right=Float32(d["x_right"]),
    n_per_region=spec.n_per_region,maxiters=spec.maxiters,batch_size=Int(t["batch_size"]),
    checkpoint_interval=Int(t["checkpoint_epochs"]),metrics_interval=Int(t["metrics_interval"]),input_encoding=Symbol(m["input_encoding"]))
end

function _run_one_gen_job!(board::TUI.GPUBoard, job::GenJob; slot::Int=job.device_id)
  mkpath(job.output_dir)
  kwargs = config_keywords(job.spec)
  cfg = Experiments.ExperimentConfig(; kwargs...)
  tag = string(job.spec.name, "_", job.spec.variant)
  label = isempty(job.spec.group) ? tag : string(tag, " [", job.spec.group, "]")
  TUI.update!(board, slot; variant=label)

  if job.spec.name in GEOMETRY_EXPERIMENTS
    geometry_cfg = cfg
    geometry_root = job.spec.config === nothing ? dirname(job.output_dir) : job.output_dir
    result = run_geometry(geometry_cfg, job.spec.name;
      output_root=geometry_root, ng=job.spec.Ng)
    cp(joinpath(geometry_root, String(job.spec.name), String(cfg.representation), "panelset.json"),
       joinpath(job.output_dir, "panelset.json"); force=true)
    TUI.update!(board, slot; variant=string(label, " ✓"))
    return result
  end

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
  elseif job.spec.name in (:gen_radius, :gen_radius_family)
    Experiments.run_gen_radius(cfg, tag;
                               output_root=job.output_dir, Ng=job.spec.Ng,
                               mode=job.spec.name === :gen_radius_family ? :family : :disk)
  else
    error("Internal: spec.name :$(job.spec.name) should have been validated at construction")
  end

  # Save the PanelSet next to the experiment's training artifacts so the
  # viewer can load it the same way it loads data/shared_*.json.
  out_path = joinpath(job.output_dir, "panelset.json")
  Experiments.save_experiment(out_path, ps)

  TUI.update!(board, slot; variant=string(label, " ✓"))
  return ps
end

end # module GeneralizationVariants