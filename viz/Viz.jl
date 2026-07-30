# ===================================================================
# Viz  —  interactive PINN diagnostic dashboard, backed by GLMakie.
#
# This module ties together the colour-theme system (Theme.jl) and
# the ODE-results viewer (NNViewer.jl) into a single, self-contained
# diagnostic tool.
#
# Quick-start:
#     julia --project=. explore.jl
#
# Or from a Julia REPL:
#     include("viz/Viz.jl")
#     using .Viz
#     Viz.explore("results/run-adam-06-03-26/training_results.json",
#                 "results/run-adam-06-03-26/loss.csv")
# ===================================================================

module Viz

include("Theme.jl")
using .Theme

include("NNViewer.jl")
using .NNViewer

"""
    explore(results_json::String, loss_csv::String;
            theme::Theme.Theme = Theme.DARK_THEME, kwargs...)

Launch the interactive GLMakie diagnostic dashboard for a PINN
training run.

# Arguments
- `results_json` — path to `training_results.json`
- `loss_csv`     — path to `loss.csv`

# Keyword arguments
- `theme`        — `Theme` (from `Viz.Theme`) controlling figure styling
- `x_range`      — evaluation domain `(lo, hi)` (default `(0, 1)`)
- `num_points`   — number of x-collocation points (default `1000`)
- `initial_start`— start iteration for the range slider (default auto)
- `initial_end`  — end   iteration for the range slider (default `1000`)
"""
function explore(results_json::String, loss_csv::String;
                 theme::Theme.Theme = Theme.DARK_THEME, kwargs...)
    NNViewer.view(results_json, loss_csv; theme = theme, kwargs...)
end

end # module Viz
