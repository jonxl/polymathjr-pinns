#=
Shared extrapolation experiment — runs for BOTH representations.
Per-region extrapolation: train in-box, test on concentric shells
R=1..Rmax of the same region. Error-vs-R curves.
=#

include("../../utils/experiments.jl")
using .Experiments
import Random; Random.seed!(1234)

isdir("data") || mkpath("data")

const REGIONS = [:saddle, :stable_node, :unstable_node,
                 :stable_spiral, :unstable_spiral, :center]

for rep in [:power_series, :eigenvalue]
    cfg = ExperimentConfig(; representation=rep)
    @info "Running extrapolate: $rep"
    ps = run_extrapolate(cfg, REGIONS, 4, "shared_extrapolate")
    path = save_experiment("data/shared_extrapolate_$(rep).json", ps)
    @info "Saved: $path"
end

@info "Extrapolate experiment complete."
