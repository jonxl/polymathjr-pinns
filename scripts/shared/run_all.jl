#=
Batch experiment runner — runs all 5 diagnostic shapes for BOTH representations.
Each shape uses the ExperimentConfig defaults (maxiters=3000, n_per_region=200, N=20).

Toggle individual experiments below. Results land in data/shared_*.json,
loadable by the viz viewer (viz/PanelViewer.jl).

Estimated time on CPU (all experiments, both reps):
  transfer:         ~15 min  (6 models × 2 reps)
  extrapolate:      ~10 min  (6 models × 2 reps)
  range:            ~12 min  (6 models × 2 reps)
  showcase:          ~5 min  (1 model × 2 reps)
  gen_radius disk:   ~3 min  (1 model × 2 reps)
  gen_radius family: ~12 min (6 models × 2 reps)
  ─────────────────────────
  Total:            ~60 min
=#

include("../../utils/experiments.jl")
using .Experiments
import Random; Random.seed!(1234)

const REGIONS = [:saddle, :stable_node, :unstable_node,
                 :stable_spiral, :unstable_spiral, :center]
const MAXITERS = 1_000_000
const CHECKPOINT_INTERVAL = 100_000
const OUTPUT_ROOT = "results/dual-rep-adam-iter-1m"
const DATA_DIR = joinpath(OUTPUT_ROOT, "data")

mkpath(DATA_DIR)

# ── Toggle which experiments to run ──
const RUN = Dict(
    :transfer           => true,
    :extrapolate        => false,
    :range              => false,
    :showcase           => false,
    :gen_radius_disk    => false,
    :gen_radius_family  => false,
)

for rep in [:power_series, :eigenvalue]
    cfg = ExperimentConfig(; representation=rep, maxiters=MAXITERS,
                           checkpoint_interval=CHECKPOINT_INTERVAL)
    println("\n" * "="^60)
    println("  Representation: $rep")
    println("="^60)

    if RUN[:transfer]
        @info "Transfer: $rep"
        ps = run_transfer(cfg, REGIONS, REGIONS, "batch_transfer";
                          rng_seed=1234, output_root=OUTPUT_ROOT)
        save_experiment(joinpath(DATA_DIR, "batch_transfer_$(rep).json"), ps)
    end

    if RUN[:extrapolate]
        @info "Extrapolate: $rep"
        ps = run_extrapolate(cfg, REGIONS, 6, "batch_extrapolate";
                             output_root=OUTPUT_ROOT)
        save_experiment(joinpath(DATA_DIR, "batch_extrapolate_$(rep).json"), ps)
    end

    if RUN[:range]
        @info "Cross-region range: $rep"
        ps = run_cross_region_range(cfg, REGIONS, REGIONS, 6, "batch_range";
                                    output_root=OUTPUT_ROOT)
        save_experiment(joinpath(DATA_DIR, "batch_range_$(rep).json"), ps)
    end

    if RUN[:showcase]
        @info "Showcase: $rep"
        ps = run_showcase(cfg, "batch_showcase"; output_root=OUTPUT_ROOT)
        save_experiment(joinpath(DATA_DIR, "batch_showcase_$(rep).json"), ps)
    end

    if RUN[:gen_radius_disk]
        @info "Gen radius disk: $rep"
        ps = run_gen_radius(cfg, "batch_genradius"; mode=:disk, Ng=81,
                            output_root=OUTPUT_ROOT)
        save_experiment(joinpath(DATA_DIR, "batch_genradius_disk_$(rep).json"), ps)
    end

    if RUN[:gen_radius_family]
        fam_cfg = ExperimentConfig(; representation=rep, n_per_region=100,
                                   maxiters=MAXITERS,
                                   checkpoint_interval=CHECKPOINT_INTERVAL)
        @info "Gen radius family: $rep"
        ps = run_gen_radius(fam_cfg, "batch_genradius"; mode=:family, Ng=81,
                            output_root=OUTPUT_ROOT)
        save_experiment(joinpath(DATA_DIR, "batch_genradius_family_$(rep).json"), ps)
    end
end

@info "Batch experiment complete." output_root=OUTPUT_ROOT
