#!/usr/bin/env julia
# ===================================================================
# explore.jl  —  PINNs diagnostic dashboard entry point
#
# Usage:
#     julia --project=. explore.jl
#     julia --project=. explore.jl results/run-adam-06-03-26/training_results.json results/run-adam-06-03-26/loss.csv
#     julia --project=. explore.jl --theme light
# ===================================================================

using GLMakie
using JSON

include("../viz/Viz.jl")
using .Viz

function is_panelset_json(path)
    isfile(path) || return false
    try
        d = JSON.parsefile(path)
        return haskey(d, "panels") && haskey(d, "name")
    catch
        return false
    end
end

function main()
    # ---- resolve paths from CLI args ---------------------------------
    results_json = nothing
    loss_csv     = nothing
    theme_name   = "dark"
    panel_paths  = String[]

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--theme" && i < length(ARGS)
            i += 1
            theme_name = ARGS[i]
        elseif arg == "--help" || arg == "-h"
            println("""
            explore.jl  —  PINNs ODE Training Diagnostic Dashboard

            Usage:
              julia --project=. explore.jl [RESULTS_JSON] [LOSS_CSV] [--theme THEME]
              julia --project=. explore.jl data/shared_transfer_power_series.json data/shared_transfer_eigenvalue.json

            Arguments:
              RESULTS_JSON   path to training_results.json (optional; auto-detected)
              LOSS_CSV       path to loss.csv (optional; auto-detected)
              PANEL_JSON      one or more experiment PanelSet JSON files emitted by scripts/shared/
              --theme NAME   colour theme: dark (default), light, high_contrast

            Examples:
              julia --project=. explore.jl
              julia --project=. explore.jl results/run-adam-06-03-26/training_results.json results/run-adam-06-03-26/loss.csv
              julia --project=. explore.jl data/shared_genradius_family_power_series.json data/shared_genradius_family_eigenvalue.json
              julia --project=. explore.jl --theme light
            """)
            return
        elseif !startswith(arg, "--")
            if is_panelset_json(arg)
                push!(panel_paths, arg)
            elseif results_json === nothing
                results_json = arg
            elseif loss_csv === nothing
                loss_csv = arg
            end
        end
        i += 1
    end

    theme = Viz.get_theme(theme_name)
    if !isempty(panel_paths)
        println("Theme:  $(theme.name)")
        println("Panels: $(join(panel_paths, ", "))")
        return Viz.explore_panels(panel_paths; theme = theme)
    end

    # ---- auto-detect latest run if paths not given -------------------
    if results_json === nothing || loss_csv === nothing
        results_dir = "results"
        if !isdir(results_dir)
            error("No '$results_dir/' directory found. Run training first, or specify paths explicitly.")
        end

        runs = sort(filter(d -> startswith(d, "run-"), readdir(results_dir)); rev = true)
        if isempty(runs)
            error("No training runs found in '$results_dir/'. Run training first.")
        end

        for run in runs
            candidate_json = joinpath(results_dir, run, "training_results.json")
            candidate_csv  = joinpath(results_dir, run, "loss.csv")
            if isfile(candidate_json)
                if results_json === nothing
                    results_json = candidate_json
                end
                if loss_csv === nothing && isfile(candidate_csv)
                    loss_csv = candidate_csv
                end
                if results_json !== nothing && loss_csv !== nothing
                    break
                end
            end
        end
    end

    if results_json === nothing
        error("Could not find training_results.json. Specify path explicitly.")
    end
    if loss_csv === nothing
        @warn "No loss.csv found — loss plots will be empty."
        loss_csv = ""   # will be caught by NNViewer
    end

    println("Theme:    $(theme.name)")
    println("Results:  $results_json")
    println("Loss CSV: $loss_csv")

    return Viz.explore(results_json, loss_csv; theme = theme)
end

fig = main()

if !isinteractive() && fig !== nothing
    while GLMakie.isopen(fig.scene)
        sleep(0.5)
    end
end
