#=
Shared transfer experiment — runs for BOTH representations.
Calls `run_transfer` from experiments.jl, which dispatches on
representation via evaluate_on, producing identical diagnostics
(heatmaps + bar chart) in both coefficient and solution space.
=#

include("../../utils/experiments.jl")
using .Experiments
import Random; Random.seed!(1234)

isdir("data") || mkpath("data")

const REGIONS = [:saddle, :stable_node, :unstable_node,
                 :stable_spiral, :unstable_spiral, :center]

for rep in [:power_series, :eigenvalue]
    cfg = ExperimentConfig(; representation=rep)
    @info "Running transfer experiment: $rep"
    ps = run_transfer(cfg, REGIONS, REGIONS, "shared_transfer"; rng_seed=1234)
    path = save_experiment("data/shared_transfer_$(rep).json", ps)
    @info "Saved: $path"
end

@info "Transfer experiment complete — both representations saved."
