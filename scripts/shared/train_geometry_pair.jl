#!/usr/bin/env julia
# Resume at completed representation boundaries; never overwrite partial training.
length(ARGS) == 3 || error("Usage: train_geometry_pair.jl OUTPUT_ROOT EXPERIMENT ITERATIONS")
using PINNStudies
using PINNStudies: GeometryExperiments
using .GeometryExperiments
const GE = GeometryExperiments
const GG = GE.GeneralizationGeometry
root, name, iterations = ARGS[1], Symbol(ARGS[2]), parse(Int, ARGS[3])
name in GG.EXPERIMENTS || error("Unknown experiment: $name")
iterations > 0 || error("Iterations must be positive")
for rep in (:power_series, :eigenvalue, :power_series_variation)
    cfg = GE.Experiments.ExperimentConfig(; representation=rep, N=20,
        n_per_region=GG.jeet_controls(name).n, maxiters=iterations)
    dir = joinpath(root, String(name), String(rep))
    if isfile(joinpath(dir, "panelset.json")) && isfile(joinpath(dir, "results.json"))
        # Reuse only runs whose full shared control contract still agrees.
        digest = GE.ensure_contract(joinpath(root, String(name)),
            GG.design(name; n=cfg.n_per_region, seed=cfg.seed), cfg, 200)
        panels = GE.load_experiment(joinpath(dir, "panelset.json"))
        get(panels.meta, "design_sha256", nothing) == digest || error("Saved controls differ: $dir")
        @info "Reusing completed geometry" name rep dir
    else
        GE.run_geometry(cfg, name; output_root=root)
    end
end
