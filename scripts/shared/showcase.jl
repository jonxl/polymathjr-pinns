#=
Shared showcase experiment — runs for BOTH representations.
Train on exponential family, test 6 scenarios (memorize, in-family,
out-of-range×2, out-of-family×2). Solution-curve overlays + rel-L2 bar chart.
=#

include("../../utils/experiments.jl")
using .Experiments
import Random; Random.seed!(1234)

isdir("data") || mkpath("data")

for rep in [:power_series, :eigenvalue]
    cfg = ExperimentConfig(; representation=rep)
    @info "Running showcase: $rep"
    ps = run_showcase(cfg, "shared_showcase")
    path = save_experiment("data/shared_showcase_$(rep).json", ps)
    @info "Saved: $path"
end

@info "Showcase experiment complete."
