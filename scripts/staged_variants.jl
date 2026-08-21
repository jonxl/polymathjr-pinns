# =============================================================================
# scripts/staged_variants.jl — entry point for the staged variant runner.
#
# Usage:
#     julia --project=. -t auto scripts/staged_variants.jl
#
# Flags:
#     --specs power_series:eigenvalue        which representations to stage
#     --neurons 64                            hidden width
#     --maxiters 3000                         Adam iterations per variant
#     --n-per-region 200                       ODEs per region in shared dataset
#     --output-root results/staged            output directory
#     --seed 1234                             base RNG seed
#
# The default spec list runs four power-series variants (N=20,25,30,35) plus
# the eigenvalue variant — five jobs total. On a five-GPU box each GPU gets
# exactly one variant; on a smaller box they queue round-robin.
# =============================================================================

using ArgParse
using Dates
using JSON
using Random

include("../utils/plugboard.jl")
using .Plugboard

include("../utils/variants.jl")
using .Variants

function parse_commandline()
  s = ArgParseSettings(description="Staged PINN variant runner — per-GPU dispatch via TUI.GPUBoard")

  @add_arg_table! s begin
    "--specs"
      help = "Comma-separated representations to run (power_series, eigenvalue)"
      arg_type = String
      default = "power_series,eigenvalue"
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
      default = joinpath("results", Dates.format(now(), "mm-dd-yy-HHMM"))
    "--seed"
      help = "Base RNG seed for dataset + per-variant seeds"
      arg_type = Int
      default = 1234
  end
  return parse_args(s)
end

parsed = parse_commandline()

# ---- Spec list ----
function build_specs(args)
    specs = VariantSpec[]
    wanted = Set{Symbol}(Symbol(strip(s)) for s in split(args["specs"], ",") if !isempty(s))
    if :power_series in wanted
        for N in (20, 25, 30, 35)
            push!(specs, VariantSpec(string("ps_N", N), :power_series;
                N=N, neuron_count=args["neurons"], maxiters=args["maxiters"],
                seed_offset=N))
        end
    end
    if :eigenvalue in wanted
        push!(specs, VariantSpec("eig", :eigenvalue;
            N=20, neuron_count=args["neurons"], maxiters=args["maxiters"],
            seed_offset=9999))
    end
    return specs
end

specs = build_specs(parsed)

@info "Staged variant run" specs=specs output_root=parsed["output-root"] n_devices=length(device_names())

# ---- Generate shared dataset ----
mkpath(parsed["output-root"])
dataset = generate_shared_dataset(n_per_region=parsed["n-per-region"],
                                  num_terms=21,
                                  seed=parsed["seed"])

# ---- Build jobs + write manifest ----
jobs = build_jobs(specs, dataset, parsed["output-root"])
manifest_path = joinpath(parsed["output-root"], "manifest.json")
write_manifest(manifest_path, jobs, Dict(
    "n_per_region" => parsed["n-per-region"],
    "num_terms" => 21,
    "seed" => parsed["seed"],
    "regions" => String.(Plugboard.TRACE_DET_REGIONS),
); n_devices=length(device_names()))
@info "Manifest written" path=manifest_path n_jobs=length(jobs)

# ---- Run ----
run_staged_variants(jobs)

@info "Staged variant run complete" output_root=parsed["output-root"] manifest=manifest_path