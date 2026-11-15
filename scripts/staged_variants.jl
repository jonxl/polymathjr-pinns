# =============================================================================
# scripts/staged_variants.jl — entry point for the staged variant runner.
#
# Usage:
#     julia --project=. -t auto scripts/staged_variants.jl
#
# Flags:
#     --specs power_series,eigenvalue        which representations to stage
#     --degrees 20,25,30,35                   power-series degrees N
#     --weightings uniform,descending         power-series coefficient weightings
#     --lanes 0                               jobs at once per device (0 = automatic)
#     --neurons 64                            hidden width
#     --maxiters 3000                         Adam iterations per variant
#     --n-per-region 200                       ODEs per region in shared dataset
#     --output-root results/staged            output directory
#     --seed 1234                             base RNG seed
#
# The default grid is every (degree, weighting) pair for the power series plus
# one eigenvalue variant — nine jobs. Eigenvalue does not depend on the degree,
# so adding a degree adds only power-series jobs.
#
# One dataset is generated for the whole run, carrying coefficients through the
# largest degree; each variant supervises a prefix of the same rows.
#
# Jobs are assigned to devices round-robin, and each device trains `--lanes`
# of them at once. Launch with enough Julia threads (-t auto) to cover the lanes.
# =============================================================================

using ArgParse
using Dates
using JSON
using Random

include("../utils/plugboard.jl")
using .Plugboard

using PINNStudies
using PINNStudies: Variants, RunManagement
using .Variants

function parse_commandline()
  s = ArgParseSettings(description="Staged PINN variant runner — per-GPU dispatch via TUI.GPUBoard")

  @add_arg_table! s begin
    "--specs"
      help = "Comma-separated representations to run (power_series, eigenvalue)"
      arg_type = String
      default = "power_series,eigenvalue"
    "--degrees"
      help = "Comma-separated power-series degrees N"
      arg_type = String
      default = "20,25,30,35"
    "--weightings"
      help = "Comma-separated power-series coefficient weightings (uniform, descending)"
      arg_type = String
      default = "uniform,descending"
    "--lanes"
      help = "Jobs trained at once per device; 0 picks one per Julia thread on GPU, one on CPU"
      arg_type = Int
      default = 0
    "--neurons"
      help = "Hidden width per variant"
      arg_type = Int
      default = 64
    "--maxiters"
      help = "Adam iterations per variant"
      arg_type = Int
      default = 3000
    "--n-per-region"
      help = "ODEs sampled per trace-determinant region (shared dataset)"
      arg_type = Int
      default = 200
    "--output-root"
      help = "Root directory for staged results; build_jobs adds a 'staged-<tag>' subdir per variant"
      arg_type = String
      default = nothing
    "--seed"
      help = "Base RNG seed for dataset + per-variant seeds"
      arg_type = Int
      default = 1234
  end
  return parse_args(s)
end

parsed = parse_commandline()

# ---- Spec list ----
symbols(text) = Symbol[Symbol(strip(t)) for t in split(text, ",") if !isempty(strip(t))]

specs = staged_specs(
    representations=symbols(parsed["specs"]),
    degrees=[parse(Int, strip(t)) for t in split(parsed["degrees"], ",") if !isempty(strip(t))],
    weightings=symbols(parsed["weightings"]),
    neuron_count=parsed["neurons"], maxiters=parsed["maxiters"])
num_terms = required_terms(specs)

parsed["lanes"] >= 0 || error("--lanes must be 0 (automatic) or a positive integer")
lanes = parsed["lanes"] == 0 ? auto_lanes(length(specs)) : parsed["lanes"]
lanes <= Threads.nthreads() || @warn "More lanes than Julia threads; jobs will share threads" lanes threads=Threads.nthreads()

@info "Staged variant run" variants=[s.name for s in specs] num_terms=num_terms lanes=lanes threads=Threads.nthreads() output_root=parsed["output-root"] n_devices=length(device_names())

# ---- Generate shared dataset ----
run_context=RunManagement.create_run(Dict("launch"=>Dict(k=>v for (k,v) in parsed if v !== nothing)); output=parsed["output-root"])
parsed["output-root"]=run_context.root
dataset = generate_shared_dataset(n_per_region=parsed["n-per-region"],
                                  num_terms=num_terms,
                                  seed=parsed["seed"])

# ---- Build jobs + write manifest ----
jobs = build_jobs(specs, dataset, parsed["output-root"])
manifest_path = joinpath(parsed["output-root"], "jobs.json")
write_manifest(manifest_path, jobs, Dict(
    "n_per_region" => parsed["n-per-region"],
    "num_terms" => num_terms,
    "lanes" => lanes,
    "seed" => parsed["seed"],
    "regions" => String.(Plugboard.TRACE_DET_REGIONS),
); n_devices=length(device_names()))
@info "Manifest written" path=manifest_path n_jobs=length(jobs)

# ---- Run ----
RunManagement.with_run_logging(run_context) do
    run_staged_variants(jobs; run=run_context, lanes=lanes)
end

@info "Staged variant run complete" output_root=parsed["output-root"] manifest=manifest_path