# =============================================================================
# scripts/generalization_variants.jl — entry point for the generalization
# experiments × representations runner.
#
# Same pattern as scripts/staged_variants.jl: one task per (experiment, rep)
# combination, dispatched round-robin across visible CUDA devices, rendered
# through the GPUBoard. The five diagnostic shapes (transfer, extrapolate,
# range, showcase, gen_radius) all run their power_series and eigenvalue
# variants side-by-side.
#
# Usage:
#     julia --project=. -t auto scripts/generalization_variants.jl
#
# Flags:
#     --reps power_series,eigenvalue          which representations
#     --experiments transfer,extrapolate      comma-separated subset (default: all 5)
#     --Rmax 6                                extrapolation shells
#     --Ng 31                                 grid size for gen_radius
#     --output-root results/gen                root for outputs
#     --n-per-region 200                       ODEs per region (forwarded to ExperimentConfig)
#     --maxiters 3000                          Adam iterations per PINN
# =============================================================================

using ArgParse
using CUDA
using Dates
using JSON

using PINNStudies
using PINNStudies: GeneralizationVariants
using .GeneralizationVariants
using PINNStudies: ExperimentConfiguration, RunManagement

function parse_commandline()
  s = ArgParseSettings(description="Generalization experiments × representations — per-GPU dispatch via TUI.GPUBoard")

  @add_arg_table! s begin
    "--config"
      help = "TOML experiment preset; explicit CLI flags override it"
      arg_type = String
    "--date"
      help = "Date directory for this run"
      arg_type = String
      default = string(Dates.today())
    "--report"
      help = "Export the HTML report into this run after training"
      action = :store_true
    "--variants", "--reps"
      dest_name = "reps"
      help = "Comma-separated presets: power_series_standard, power_series_descending, eigenvalue_standard; legacy names accepted"
      arg_type = String
      default = nothing
    "--experiments"
      help = "all, or comma-separated experiments: transfer, extrapolate, range, showcase, gen_radius, gen_radius_family, family_disks, spiral_transfer, pair_transfer, parabola_segment, parabola_slide, parabola_curves, parabola_bias, axis_transfer, axis_sweep"
      arg_type = String
      default = nothing
    "--Rmax"
      help = "Number of extrapolation shells (extrapolate, range)"
      arg_type = Int
      default = 6
    "--Ng"
      help = "Grid resolution; defaults to each geometry experiment’s reference setting"
      arg_type = Int
      default = nothing
    "--n-per-region"
      help = "ODEs per patch/region; defaults to each geometry experiment’s reference count"
      arg_type = Int
      default = nothing
    "--maxiters"
      help = "Adam iterations per PINN (forwarded to Experiments.ExperimentConfig)"
      arg_type = Int
      default = nothing
    "--degrees"
      help = "Comma-separated power-series degrees N; overrides the preset. Several values run a degree sweep"
      arg_type = String
      default = nothing
    "--lanes"
      help = "Experiment jobs run at once per device"
      arg_type = Int
      default = 1
    "--dry-run"
      help = "Print all planned jobs without training or writing results"
      action = :store_true
    "--output-root"
      help = "Root directory for staged results; build_gen_jobs adds a 'gen-<exp>-<rep>' subdir per job"
      arg_type = String
      default = nothing
  end
  return parse_args(s)
end

parsed = parse_commandline()
base_config = ExperimentConfiguration.load_config(parsed["config"])
parsed["maxiters"] = something(parsed["maxiters"], base_config["training"]["maxiters"])
base_config["training"]["maxiters"] = parsed["maxiters"]
parsed["experiments"] = something(parsed["experiments"],base_config["run"]["experiments"])
base_config["run"]["experiments"] = parsed["experiments"]
if parsed["reps"] === nothing
    parsed["reps"] = parsed["config"] === nothing ? "power_series,eigenvalue,power_series_variation" :
      (base_config["model"]["representation"] == "power_series" && base_config["objective"]["coefficient_weighting"] == "descending" ? "power_series_variation" : base_config["model"]["representation"])
end
parsed["maxiters"] > 0 || error("maxiters must be positive")
degrees = parsed["degrees"] === nothing ? Int[base_config["model"]["degree"]] :
    [parse(Int, strip(t)) for t in split(parsed["degrees"], ",") if !isempty(strip(t))]
!isempty(degrees) && allunique(degrees) && all(d -> 1 <= d <= 40, degrees) || error("degrees must be distinct values in 1:40")
sweep = length(degrees) > 1
# The eigenvalue model has no series degree. Its jobs keep the preset's value, which only
# sets how many coefficients its stored equations carry and never enters its training.
reference_degree = base_config["model"]["degree"]
sweep || (base_config["model"]["degree"] = only(degrees))
parsed["lanes"] >= 1 || error("lanes must be a positive integer")
parsed["lanes"] <= Threads.nthreads() || @warn "More lanes than Julia threads; jobs will share threads" lanes=parsed["lanes"] threads=Threads.nthreads()
if parsed["n-per-region"] === nothing && base_config["data"]["n_per_region"] > 0
    parsed["n-per-region"] = base_config["data"]["n_per_region"]
end
function job_config(rep, experiment, degree)
    c=deepcopy(base_config)
    c["model"]["degree"]=degree
    c["model"]["representation"]=String(ExperimentConfiguration.canonical_representation(rep))
    c["objective"]["coefficient_weighting"]=String(ExperimentConfiguration.coefficient_weighting(rep))
    if experiment in GeneralizationVariants.GEOMETRY_EXPERIMENTS || rep === :eigenvalue
        c["model"]["input_encoding"]="trace_determinant"
    end
    return c
end

# ---- Parse reps / experiments ----
function parse_reps(s::AbstractString)
  reps = Symbol[]
  for tok in split(s, ",")
    t = strip(tok)
    isempty(t) && continue
    sym = get(Dict("power_series_standard"=>:power_series,"power_series_descending"=>:power_series_variation,"eigenvalue_standard"=>:eigenvalue),t,Symbol(t))
    sym in (:power_series, :eigenvalue, :power_series_variation) || error(
      "Unknown representation :$sym; expected power_series, eigenvalue or power_series_variation")
    push!(reps, sym)
  end
  return Tuple(reps)
end

function parse_experiments(s::AbstractString)
  s == "all" && return GeneralizationVariants.ALL_EXPERIMENTS
  exps = Symbol[]
  valid = GeneralizationVariants.ALL_EXPERIMENTS
  for tok in split(s, ",")
    t = strip(tok)
    isempty(t) && continue
    sym = Symbol(t)
    sym in valid || error("Unknown experiment :$sym; expected one of $valid")
    push!(exps, sym)
  end
  return Tuple(exps)
end

reps = parse_reps(parsed["reps"])
!isempty(reps) && length(unique(reps)) == length(reps) || error("Select at least one distinct variant")
experiments = parse_experiments(parsed["experiments"])
!isempty(experiments) && length(unique(experiments)) == length(experiments) || error("Select at least one distinct experiment")

# ---- Visible device count ----
function visible_device_count()
  if CUDA.functional()
    return CUDA.ndevices()
  else
    return 1
  end
end

@info "Generalization variants run" reps=reps experiments=experiments Rmax=parsed["Rmax"] Ng=parsed["Ng"] n_devices=visible_device_count()

# ---- Build spec list ----
# Each entry is (variant, degree, folder). Without a sweep there is one flat entry per
# variant. In a sweep, power-series variants get one entry per degree and the eigenvalue
# model, which does not depend on the degree, gets a single shared entry.
plan = Tuple{Symbol,Int,String}[]
for rep in reps
    if !sweep
        push!(plan, (rep, only(degrees), ""))
    elseif rep === :eigenvalue
        push!(plan, (rep, reference_degree, ""))
    else
        append!(plan, [(rep, N, "degree-$N") for N in degrees])
    end
end
specs = [GeneralizationSpec(exp,rep;
    Rmax=parsed["Rmax"],
    Ng=something(parsed["Ng"], exp in GeneralizationVariants.GEOMETRY_EXPERIMENTS ?
      GeneralizationVariants.GeometryExperiments.GeneralizationGeometry.jeet_controls(exp).ng : 31),
    n_per_region=parsed["n-per-region"], maxiters=parsed["maxiters"],config=job_config(rep,exp,degree),group=group)
    for (rep,degree,group) in plan for exp in experiments]

# Resolve experiment-dependent defaults before persisting or displaying the configuration.
for spec in specs
  spec.config["data"]["n_per_region"] = spec.n_per_region
end

# ---- Build jobs + run ----
if parsed["dry-run"]
  for spec in specs
    println(spec.name, " / ", spec.representation, " / weighting=", spec.coefficient_weighting, " / iterations=", spec.maxiters,
            " / points_per_patch_or_region=", spec.n_per_region,
            spec.representation === :eigenvalue ? "" : string(" / degree=", spec.config["model"]["degree"]))
  end
  println("Total experiment/variant jobs: ", length(specs))
  println("Power-series degrees: ", join(degrees, ","), "; jobs at once per device: ", parsed["lanes"])
  sweep && :eigenvalue in reps && println("Eigenvalue model: trained once; it has no series degree")
  exit(0)
end
# Persist every resolved job, including geometry-specific defaults.
base_config["jobs"] = [Dict("experiment"=>String(s.name),"variant"=>String(s.variant),
  "regions"=>String.(s.regions),"Rmax"=>s.Rmax,"Ng"=>s.Ng,"n_per_region"=>s.n_per_region,"group"=>s.group,"config"=>s.config) for s in specs]
run = RunManagement.create_run(base_config; root=base_config["run"]["root"], date=Date(parsed["date"]), output=parsed["output-root"])
jobs = build_gen_jobs(specs,joinpath(run.root,"jobs"))
println("Training output: ",run.root)
println("Report directory: ",joinpath(run.root,"report"))
failed=false
RunManagement.with_run_logging(run) do
  try
    run_generalization_variants(jobs; run=run, lanes=parsed["lanes"])
  catch e
    global failed = true
    @error "Run did not complete successfully" exception=(e, catch_backtrace())
  end
  if parsed["report"]
    python = get(ENV, "STUDY_PYTHON", "")
    exporter = isempty(python) ? `uv run` : Cmd([python])
    try
      Base.run(`$exporter scripts/charts/export_dual_study.py --date $(parsed["date"]) --output $(joinpath(run.root,"report")) --geometry-root $(joinpath(run.root,"jobs"))`)
      run.manifest["report_status"] = "completed"
    catch e
      global failed = true
      run.manifest["report_status"] = "failed"
      @error "Report export failed; training artifacts retained" exception=(e, catch_backtrace())
    end
    RunManagement.persist!(run)
  end
end
println("Run status: ",run.manifest["status"],"; output: ",run.root)
failed && exit(1)
