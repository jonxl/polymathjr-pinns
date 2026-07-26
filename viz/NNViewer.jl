# Interactive PINN ODE training results visualizer, backed by GLMakie.
#
# view(results_json, loss_csv) opens one window with:
#
#   - 4x2 grid of diagnostic plots: solution comparison, solution error,
#     coefficient comparison, coefficient error, total/bc/pde/supervised loss.
#   - A slider that scrubs through training iterations — non-loss plots
#     update from the nearest milestone snapshot; loss curves show the
#     full history with a vertical indicator at the current iteration.
#
# This replaces the Python nn-viewer (matplotlib + PyQt5) with a
# native Julia + GLMakie implementation.
#
# Every Makie name is qualified with GLMakie. — Plots (loaded by
# viz/Viz.jl) exports clashing names like heatmap!/lines!.
#
# Run from a Julia session that has GLMakie available:
#   include("pinns/viz/NNViewer.jl")
#   NNViewer.view("results/training_results.json", "results/loss.csv")

module NNViewer

import GLMakie
using JSON
using CSV
using DataFrames
using Printf

# Palette (matching the Python nn-viewer colours for familiarity)
const BENCHMARK_COLOR = "#4fc3f7"
const PINN_COLOR      = "#ff8a65"
const ANALYTICAL_COLOR = "#66bb6a"
const ERROR_COLOR     = "#ef5350"
const LOSS_COLORS = Dict(
    "total"      => "#e0e0e0",
    "bc"         => "#ef5350",
    "pde"        => "#42a5f5",
    "supervised" => "#66bb6a",
)
const MARKER_COLOR = "#ffeb3b"

# ---- factorial power-series evaluation -----------------------------------
#
# u(x) = Σ coeffᵢ · xⁱ⁻¹ / (i−1)!   (i ∈ 1:N)
#
# Equivalent to the Python version:
#     sum(coeff[i] * x^i / factorial(i))   (i ∈ 0:N−1)
# because Julia is 1-indexed and Python 0-indexed, and both divide by
# the same factorial value for the corresponding term.
# --------------------------------------------------------------------------
function evaluate_factorial_power_series(coeffs::Vector{Float64}, x::AbstractVector)
    result = zeros(Float64, length(x))
    for (i, c) in enumerate(coeffs)
        result .+= c .* (x .^ (i - 1)) ./ factorial(i - 1)
    end
    return result
end

# ---- public entry point --------------------------------------------------

"""
    view(results_json_path::String, loss_csv_path::String;
         x_range=(0.0, 1.0), num_points=1000, initial_iteration=1000)

Open an interactive GLMakie window for exploring PINN training results.

`results_json_path` should point to a `training_results.json` file
(metadata + milestones format as written by `training_schemes.jl`).
`loss_csv_path` should point to `loss.csv` (iter, total, bc, pde,
supervised columns).
"""
function view(results_json_path::String, loss_csv_path::String;
              x_range=(0.0, 1.0), num_points=1000,
              initial_iteration=1000)

    # ---- load training_results.json --------------------------------------
    data = JSON.parsefile(results_json_path)
    metadata   = data["metadata"]
    milestones = data["milestones"]
    benchmark_coeffs = Float64.(metadata["benchmark_coefficients"])
    ode_matrix       = Float64.(metadata["ode_matrix"])

    # iteration → snapshot lookup
    iter_to_snapshot = Dict{Int,Dict}()
    for m in milestones
        iter_to_snapshot[Int(m["iteration"])] = m
    end
    milestone_iters = sort(collect(keys(iter_to_snapshot)))

    # ---- load loss CSV ---------------------------------------------------
    df = CSV.read(loss_csv_path, DataFrame)
    loss_iters      = Vector{Int}(df[!, :iteration])
    loss_total      = Vector{Float64}(df[!, :total])
    loss_bc         = Vector{Float64}(df[!, :bc])
    loss_pde        = Vector{Float64}(df[!, :pde])
    loss_supervised = Vector{Float64}(df[!, :supervised])

    # ---- iteration range & slider step -----------------------------------
    if !isempty(milestone_iters) && !isempty(loss_iters)
        iter_min = min(milestone_iters[1], loss_iters[1])
        iter_max = max(milestone_iters[end], loss_iters[end])
    elseif !isempty(milestone_iters)
        iter_min = milestone_iters[1]
        iter_max = milestone_iters[end]
    elseif !isempty(loss_iters)
        iter_min = loss_iters[1]
        iter_max = loss_iters[end]
    else
        iter_min = 0
        iter_max = max(initial_iteration, 2000)
    end
    init_val = clamp(initial_iteration, iter_min, iter_max)
    iter_step = if length(loss_iters) > 1
        max(1, Int(round(median(diff(loss_iters)))))
    elseif length(milestone_iters) > 1
        milestone_iters[2] - milestone_iters[1]
    else
        100
    end

    # ---- precompute static evaluation data --------------------------------
    x_eval = collect(range(x_range[1], x_range[2]; length=num_points))
    benchmark_series = evaluate_factorial_power_series(benchmark_coeffs, x_eval)
    n_bench = length(benchmark_coeffs)

    # ---- helpers ----------------------------------------------------------

    # Latest milestone snapshot at or before `iteration`
    function get_snapshot(iteration)
        snap = nothing
        for it in milestone_iters
            it <= iteration && (snap = iter_to_snapshot[it])
        end
        return snap
    end

    # ---- GLMakie window ---------------------------------------------------

    GLMakie.activate!()
    fig = GLMakie.Figure(size = (1920, 1200))

    # Iteration slider
    sl = GLMakie.Slider(fig[8, 1:4];
                        range = iter_min:iter_step:iter_max,
                        startvalue = init_val)
    it_obs = sl.value

    # Title bar
    GLMakie.Label(fig[1, 1:4],
        GLMakie.lift(it -> @sprintf(
            "PINN ODE Training Analysis  —  iteration %d  (0 … %d)",
            it, iter_max), it_obs),
        fontsize = 20, font = :bold, halign = :center)

    # Reactive data driven by the slider
    snap_obs = GLMakie.lift(get_snapshot, it_obs)

    pinn_coeffs_obs = GLMakie.lift(snap_obs) do snap
        snap === nothing && return Float64[]
        return Float64.(snap["coefficients"])
    end

    pinn_series_obs = GLMakie.lift(pinn_coeffs_obs) do pc
        isempty(pc) && return zeros(length(x_eval))
        return evaluate_factorial_power_series(pc, x_eval)
    end

    # Number of coefficients to compare (min of benchmark & PINN)
    n_compare_obs = GLMakie.lift(pc -> min(n_bench, length(pc)), pinn_coeffs_obs)

    # coeff indices the PINN line will use (0 … n−1)
    coeff_idx_obs = GLMakie.lift(n -> Float64.(0:(n - 1)), n_compare_obs)

    # === Plots row 3: solution comparison + solution error ================
    ax1 = GLMakie.Axis(fig[3, 1:2];
                       title = "ODE Solution Comparison",
                       xlabel = "x", ylabel = "u(x)")
    GLMakie.lines!(ax1, x_eval, benchmark_series;
                   color = BENCHMARK_COLOR, linestyle = :dash,
                   label = "Benchmark Series", linewidth = 2)
    GLMakie.lines!(ax1, x_eval, pinn_series_obs;
                   color = PINN_COLOR, linestyle = :solid,
                   label = "PINN Series", linewidth = 2)
    GLMakie.axislegend(ax1; labelsize = 9, position = :lt)

    ax2 = GLMakie.Axis(fig[3, 3:4];
                       title = "Absolute Error of Solution",
                       xlabel = "x", ylabel = "|Error|",
                       yscale = Makie.log10)
    abs_error_obs = GLMakie.lift(ps -> abs.(benchmark_series .- ps),
                                 pinn_series_obs)
    GLMakie.lines!(ax2, x_eval, abs_error_obs;
                   color = ERROR_COLOR, linewidth = 2)

    # === Plots row 4: coefficient comparison + coefficient error ==========
    ax3 = GLMakie.Axis(fig[4, 1:2];
                       title = "Coefficient Comparison",
                       xlabel = "Coefficient Index",
                       ylabel = "Coefficient Value")
    # Benchmark (static — does not change with slider)
    bench_idx_full = Float64.(0:(n_bench - 1))
    GLMakie.lines!(ax3, bench_idx_full, benchmark_coeffs;
                   color = BENCHMARK_COLOR, linewidth = 2,
                   label = "Benchmark")
    # PINN (reactive — truncates to min length)
    pinn_y_slice_obs = GLMakie.lift(pinn_coeffs_obs, n_compare_obs) do pc, n
        n == 0 && return Float64[]
        return pc[1:n]
    end
    GLMakie.lines!(ax3, coeff_idx_obs, pinn_y_slice_obs;
                   color = PINN_COLOR, linewidth = 2,
                   label = "PINN")
    GLMakie.axislegend(ax3; labelsize = 9, position = :rt)

    ax4 = GLMakie.Axis(fig[4, 3:4];
                       title = "Coefficient Error",
                       xlabel = "Coefficient Index",
                       ylabel = "|Error|",
                       yscale = Makie.log10)
    coeff_err_obs = GLMakie.lift(pinn_coeffs_obs, n_compare_obs) do pc, n
        n == 0 && return Float64[]
        return abs.(benchmark_coeffs[1:n] .- pc[1:n])
    end
    GLMakie.lines!(ax4, coeff_idx_obs, coeff_err_obs;
                   color = ERROR_COLOR, linewidth = 2)

    # === Plots rows 5–6: loss curves (full history, log y) ================

    function loss_axis(parent, title, loss_vec, color)
        ax = GLMakie.Axis(parent;
                          title = title,
                          xlabel = "Iteration",
                          ylabel = "Loss",
                          yscale = Makie.log10)
        GLMakie.lines!(ax, loss_iters, loss_vec;
                       color = color, linewidth = 1.2)
        GLMakie.vlines!(ax, it_obs;
                        color = (MARKER_COLOR, 0.6), linewidth = 2)
        return ax
    end

    loss_axis(fig[5, 1:2], "Total Loss",      loss_total,      LOSS_COLORS["total"])
    loss_axis(fig[5, 3:4], "BC Loss",         loss_bc,         LOSS_COLORS["bc"])
    loss_axis(fig[6, 1:2], "PDE Loss",        loss_pde,        LOSS_COLORS["pde"])
    loss_axis(fig[6, 3:4], "Supervised Loss", loss_supervised, LOSS_COLORS["supervised"])

    # ---- Status bar -------------------------------------------------------
    GLMakie.Label(fig[7, 1:4],
        GLMakie.lift(snap_obs) do snap
            snap === nothing && return "No snapshot at this iteration."
            it = snap["iteration"]
            obj = snap["objective"]
            return @sprintf("Snapshot  iteration %d  |  objective = %.6e", it, obj)
        end,
        fontsize = 11, halign = :left, color = :gray70)

    # ---- Row / column sizing ---------------------------------------------
    GLMakie.rowsize!(fig.layout, 1, 36)
    GLMakie.rowsize!(fig.layout, 8, 36)
    GLMakie.colgap!(fig.layout, 16)
    GLMakie.rowgap!(fig.layout, 8)

    GLMakie.display(fig)
    return fig
end

end # module NNViewer
