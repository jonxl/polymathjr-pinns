#=
Shared cross-region range experiment — runs for BOTH representations.
Train per-region, test on every region at concentric shells R=1..Rmax.
One panel per training region: error-vs-R lines (solid=in-family, dashed=out).
=#

include("../../utils/experiments.jl")
using .Experiments
import Random; Random.seed!(1234)

isdir("data") || mkpath("data")

const REGIONS = [:saddle, :stable_node, :unstable_node,
                 :stable_spiral, :unstable_spiral, :center]

for rep in [:power_series, :eigenvalue]
    cfg = ExperimentConfig(; representation=rep)
    @info "Running cross-region range: $rep"
    ps = run_cross_region_range(cfg, REGIONS, REGIONS, 4, "shared_range")
    path = save_experiment("data/shared_range_$(rep).json", ps)
    @info "Saved: $path"
end

@info "Cross-region range experiment complete."
