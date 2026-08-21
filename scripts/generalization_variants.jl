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
using Dates
using JSON

include("../utils/generalization_variants.jl")
using .GeneralizationVariants

function parse_commandline()
  s = ArgParseSettings(description="Generalization experiments × representations — per-GPU dispatch via TUI.GPUBoard")

  @add_arg_table! s begin
    "--reps"
      help = "Comma-separated representations (power_series, eigenvalue)"
      arg_type = String
      default = "power_series,eigenvalue"
    "--experiments"
      help = "Comma-separated subset of: transfer, extrapolate, range, showcase, gen_radius"
      arg_type = String
      default = "transfer,extrapolate,range,showcase,gen_radius"
    "--Rmax"
      help = "Number of extrapolation shells (extrapolate, range)"
      arg_type = Int
      default = 6
    "--Ng"
      help = "Grid size for gen_radius"
      arg_type = Int
      default = 31
    "--n-per-region"
      help = "ODEs per region (forwarded to Experiments.ExperimentConfig)"
      arg_type = Int
      default = 200
    "--maxiters"
      help = "Adam iterations per PINN (forwarded to Experiments.ExperimentConfig)"
      arg_type = Int
      default = 3000
    "--output-root"
      help = "Root directory for staged results; build_gen_jobs adds a 'gen-<exp>-<rep>' subdir per job"
      arg_type = String
      default = joinpath("results", Dates.format(now(), "mm-dd-yy-HHMM"))
  end
  return parse_args(s)
end

parsed = parse_commandline()

# ---- Parse reps / experiments ----
function parse_reps(s::AbstractString)
  reps = Symbol[]
  for tok in split(s, ",")
    t = strip(tok)
    isempty(t) && continue
    sym = Symbol(t)
    sym in (:power_series, :eigenvalue) || error(
      "Unknown representation :$sym; expected power_series or eigenvalue")
    push!(reps, sym)
  end
  return Tuple(reps)
end

function parse_experiments(s::AbstractString)
  exps = Symbol[]
  valid = (:transfer, :extrapolate, :range, :showcase, :gen_radius)
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
experiments = parse_experiments(parsed["experiments"])

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
specs = default_generalization_specs(reps=reps, experiments=experiments,
                                     Rmax=parsed["Rmax"], Ng=parsed["Ng"],
                                     n_per_region=parsed["n-per-region"],
                                     maxiters=parsed["maxiters"])

# ---- Build jobs + run ----
mkpath(parsed["output-root"])
jobs = build_gen_jobs(specs, parsed["output-root"])

@info "Built jobs" n_jobs=length(jobs) output_root=parsed["output-root"]

run_generalization_variants(jobs)

@info "Generalization variants run complete" output_root=parsed["output-root"]