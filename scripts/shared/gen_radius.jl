#=
Shared generalization-radius experiment — runs for BOTH representations.
Two modes:
  :disk   — train on compact disk of (τ,Δ), evaluate on full plane grid
  :family — train one model per region, map error + area under ε-contour
=#

include("../../utils/experiments.jl")
using .Experiments
import Random; Random.seed!(1234)

isdir("data") || mkpath("data")

for rep in [:power_series, :eigenvalue]
    cfg = ExperimentConfig(; representation=rep)
    fam_cfg = ExperimentConfig(; representation=rep, n_per_region=100)

    @info "Running gen_radius disk: $rep"
    ps_disk = run_gen_radius(cfg, "shared_genradius"; mode=:disk, Ng=21)
    path_disk = save_experiment("data/shared_genradius_disk_$(rep).json", ps_disk)
    @info "Saved: $path_disk"

    @info "Running gen_radius family: $rep"
    ps_fam = run_gen_radius(fam_cfg, "shared_genradius"; mode=:family, Ng=21)
    path_fam = save_experiment("data/shared_genradius_family_$(rep).json", ps_fam)
    @info "Saved: $path_fam"
end

@info "Generalization-radius experiment complete."
