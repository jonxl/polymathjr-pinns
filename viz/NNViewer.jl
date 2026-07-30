# ===================================================================
# Interactive PINN ODE training-results viewer, backed by GLMakie.
#
#   view(results_json, loss_csv; theme, …)
#
# Opens one GLMakie window with:
#
#   • 4×2 grid of diagnostic plots — solution comparison, solution
#     error, coefficient comparison, coefficient error, total / BC /
#     PDE / supervised loss.
#   • Two iteration-range sliders (start, end) so loss curves are
#     windowed between start … end and the right handle selects the
#     milestone snapshot used for non-loss plots.
#
# This is the Julia + GLMakie port of the Python nn-viewer
# (matplotlib + PyQt5 → ODEResultsVisualizer / GeneralizedVisualizer).
# ===================================================================

module NNViewer

using GLMakie
using JSON
using CSV
using DataFrames
using Printf
using Statistics: median

# ---- data-series palette (matching the Python nn-viewer) -----------

const BENCHMARK_COLOR  = "#4fc3f7"
const PINN_COLOR       = "#ff8a65"
const ANALYTICAL_COLOR = "#66bb6a"
const ERROR_COLOR      = "#ef5350"
const LOSS_COLORS = Dict(
    "total"      => "#e0e0e0",
    "bc"         => "#ef5350",
    "pde"        => "#42a5f5",
    "supervised" => "#66bb6a",
)
const MARKER_COLOR = "#ffeb3b"

# ---- axis styling helper -------------------------------------------

function style_axis!(ax, theme)
    t  = GLMakie.to_color
    ax.backgroundcolor = t(theme.bg_secondary)
    ax.xgridcolor      = t(theme.grid_color)
    ax.ygridcolor      = t(theme.grid_color)
    ax.xlabelcolor     = t(theme.text_primary)
    ax.ylabelcolor     = t(theme.text_primary)
    ax.xtickcolor      = t(theme.text_primary)
    ax.ytickcolor      = t(theme.text_primary)
    ax.titlecolor      = t(theme.text_primary)
    ax.xgridvisible    = true
    ax.ygridvisible    = true
    ax.xgridwidth      = 0.5
    ax.ygridwidth      = 0.5
    return ax
end

# ---- factorial power-series evaluation -----------------------------
#
#  u(x) = Σ  coeffᵢ · xⁱ⁻¹ / (i−1)!    (i ∈ 1:N, Julia indexing)
#
# Matches the Python version:
#     sum(coeff[i] * x^i / factorial(i))   (i ∈ 0:N−1)
# -------------------------------------------------------------------

function evaluate_factorial_power_series(coeffs::Vector{Float64},
                                         x::AbstractVector)
    result = zeros(Float64, length(x))
    fact = 1.0                     # 0! = 1
    for (i, c) in enumerate(coeffs)
        result .+= c .* (x .^ (i - 1)) ./ fact
        fact *= i                   # fact = i! for the next term
    end
    return result
end

# ---- public entry point -------------------------------------------

"""
    view(results_json_path, loss_csv_path;
         theme        = DARK_THEME,
         x_range      = (0.0, 1.0),
         num_points   = 1000,
         initial_start = nothing,
         initial_end   = 1000)

Open an interactive GLMakie window for exploring PINN training results.

# Arguments
- `results_json_path` — path to `training_results.json`
  (metadata + milestones, as written by `training_schemes.jl`).
- `loss_csv_path` — path to `loss.csv`
  (iteration, total, bc, pde, supervised columns).

# Keyword arguments
- `theme`          — `VizTheme` (Theme.jl).
- `x_range`        — evaluation x-domain as `(lo, hi)`.
- `num_points`     — number of collocation points.
- `initial_start`  — initial left-handle iteration  (default: auto).
- `initial_end`    — initial right-handle iteration (default: 1000).
"""
function view(results_json_path::String, loss_csv_path::String;
              theme::Any = nothing,
              x_range = (0.0, 1.0), num_points::Int = 1000,
              initial_start = nothing,
              initial_end = 1000)

    # ---- load training_results.json ---------------------------------
    data = JSON.parsefile(results_json_path)
    metadata   = get(data, "metadata", Dict())
    milestones = get(data, "milestones", [])

    benchmark_coeffs = haskey(metadata, "benchmark_coefficients") ?
        Float64.(metadata["benchmark_coefficients"]) : Float64[]
    n_bench = length(benchmark_coeffs)

    iter_to_snapshot = Dict{Int,Dict}()
    for m in milestones
        iter_to_snapshot[Int(m["iteration"])] = m
    end
    milestone_iters = sort(collect(keys(iter_to_snapshot)))

    # ---- load loss CSV ---------------------------------------------
    loss_available = isfile(loss_csv_path)
    if loss_available
        df = CSV.read(loss_csv_path, DataFrame)
        loss_iters      = Vector{Int}(df[!, :iteration])
        loss_total      = Vector{Float64}(df[!, :total])
        loss_bc         = Vector{Float64}(df[!, :bc])
        loss_pde        = Vector{Float64}(df[!, :pde])
        loss_supervised = Vector{Float64}(df[!, :supervised])
    else
        loss_iters       = Int[]
        loss_total       = Float64[]
        loss_bc          = Float64[]
        loss_pde         = Float64[]
        loss_supervised  = Float64[]
    end

    # ---- iteration range & slider step -----------------------------
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
        iter_max = max(initial_end, 2000)
    end

    iter_step = if length(loss_iters) > 1
        max(1, Int(round(median(diff(loss_iters)))))
    elseif length(milestone_iters) > 1
        milestone_iters[2] - milestone_iters[1]
    else
        100
    end

    if initial_start === nothing
        initial_start = iter_min
    end
    init_s = clamp(initial_start, iter_min, iter_max)
    init_e = clamp(initial_end,   iter_min, iter_max)
    if init_s >= init_e
        init_s = max(iter_min, init_e - iter_step)
    end

    slider_range = iter_min:iter_step:iter_max

    # ---- precompute static evaluation data --------------------------
    x_eval = collect(range(x_range[1], x_range[2]; length = num_points))
    benchmark_series = isempty(benchmark_coeffs) ?
        zeros(length(x_eval)) :
        evaluate_factorial_power_series(benchmark_coeffs, x_eval)

    # ---- helpers ----------------------------------------------------
    function get_snapshot(iteration)
        snap = nothing
        for it in milestone_iters
            it <= iteration && (snap = iter_to_snapshot[it])
        end
        return snap
    end

    # ---- GLMakie window ---------------------------------------------
    GLMakie.activate!()
    fig = GLMakie.Figure(size = (1920, 1200))

    # ---- theme application ------------------------------------------
    c_primary    = (theme === nothing) ? :white    : GLMakie.to_color(theme.text_primary)
    c_secondary  = (theme === nothing) ? :gray70  : GLMakie.to_color(theme.text_secondary)
    if theme !== nothing
        fig.scene.backgroundcolor = GLMakie.to_color(theme.bg_primary)
    end

    # ---- iteration range slider (single bar, two handles) ------------
    sl = GLMakie.IntervalSlider(fig[8, 1:4];
                                range = slider_range,
                                startvalues = (init_s, init_e))
    interval_obs = sl.interval
    iter_s_obs = GLMakie.lift(v -> Int(round(v[1])), interval_obs)
    iter_e_obs = GLMakie.lift(v -> Int(round(v[2])), interval_obs)

    GLMakie.Label(fig[7, 1:4],
                  GLMakie.lift(iter_s_obs, iter_e_obs) do s, e
                      @sprintf("Iteration Range:  %d  —  %d", s, e)
                  end,
                  fontsize = 12, halign = :center, color = c_primary)

    # ---- title bar --------------------------------------------------
    title_str = GLMakie.lift(iter_e_obs) do it
        @sprintf("PINN ODE Training Analysis  —  iteration %d  (0 … %d)",
                 it, iter_max)
    end
    GLMakie.Label(fig[1, 1:4], title_str;
                  fontsize = 20, font = :bold, halign = :center,
                  color = c_primary)

    # ---- reactive data driven by the END slider ---------------------
    snap_obs = GLMakie.lift(get_snapshot, iter_e_obs)

    pinn_coeffs_obs = GLMakie.lift(snap_obs) do snap
        snap === nothing && return Float64[]
        return Float64.(snap["coefficients"])
    end

    pinn_series_obs = GLMakie.lift(pinn_coeffs_obs) do pc
        isempty(pc) && return zeros(length(x_eval))
        return evaluate_factorial_power_series(pc, x_eval)
    end

    n_compare_obs = GLMakie.lift(pc -> min(n_bench, length(pc)), pinn_coeffs_obs)
    coeff_idx_obs = GLMakie.lift(n -> Float64.(0:(max(0, n - 1))), n_compare_obs)

    # ---- Row 3: solution comparison + solution error ----------------
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
                       yscale = GLMakie.log10)
    abs_error_obs = GLMakie.lift(ps -> abs.(benchmark_series .- ps),
                                 pinn_series_obs)
    GLMakie.lines!(ax2, x_eval, abs_error_obs;
                   color = ERROR_COLOR, linewidth = 2)

    if theme !== nothing
        style_axis!(ax1, theme)
        style_axis!(ax2, theme)
    end

    # ---- Row 4: coefficient comparison + coefficient error ----------
    ax3 = GLMakie.Axis(fig[4, 1:2];
                       title = "Coefficient Comparison",
                       xlabel = "Coefficient Index",
                       ylabel = "Coefficient Value")
    bench_idx_full = Float64.(0:(max(0, n_bench - 1)))
    GLMakie.lines!(ax3, bench_idx_full, benchmark_coeffs;
                   color = BENCHMARK_COLOR, linewidth = 2,
                   label = "Benchmark")
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
                       yscale = GLMakie.log10)
    coeff_err_obs = GLMakie.lift(pinn_coeffs_obs, n_compare_obs) do pc, n
        n == 0 && return Float64[]
        return abs.(benchmark_coeffs[1:n] .- pc[1:n])
    end
    GLMakie.lines!(ax4, coeff_idx_obs, coeff_err_obs;
                   color = ERROR_COLOR, linewidth = 2)

    if theme !== nothing
        style_axis!(ax3, theme)
        style_axis!(ax4, theme)
    end

    # ---- Rows 5–6: loss curves (full history + window highlight) ----
    function loss_axis(parent, title, loss_vec, color)
        ax = GLMakie.Axis(parent;
                          title = title,
                          xlabel = "Iteration",
                          ylabel = "Loss",
                          yscale = GLMakie.log10)
        if !isempty(loss_vec)
            GLMakie.lines!(ax, loss_iters, loss_vec;
                           color = color, linewidth = 1.2)
        end
        GLMakie.vspan!(ax, iter_s_obs, iter_e_obs;
                       color = (MARKER_COLOR, 0.12))
        GLMakie.vlines!(ax, iter_s_obs;
                        color = (MARKER_COLOR, 0.35), linewidth = 1.5,
                        linestyle = :dash)
        GLMakie.vlines!(ax, iter_e_obs;
                        color = (MARKER_COLOR, 0.60), linewidth = 2)
        if theme !== nothing
            style_axis!(ax, theme)
        end
        return ax
    end

    if loss_available
        loss_axis(fig[5, 1:2], "Total Loss",      loss_total,      LOSS_COLORS["total"])
        loss_axis(fig[5, 3:4], "BC Loss",         loss_bc,         LOSS_COLORS["bc"])
        loss_axis(fig[6, 1:2], "PDE Loss",        loss_pde,        LOSS_COLORS["pde"])
        loss_axis(fig[6, 3:4], "Supervised Loss", loss_supervised, LOSS_COLORS["supervised"])
    else
        for (ri, ci, ttl) in [(5, 1, "Total Loss"), (5, 3, "BC Loss"),
                              (6, 1, "PDE Loss"),  (6, 3, "Supervised Loss")]
            ax = GLMakie.Axis(fig[ri, ci:(ci + 1)];
                              title = ttl, xlabel = "Iteration",
                              ylabel = "Loss", yscale = GLMakie.log10)
            GLMakie.text!(ax, 0.5, 0.5; text = "(no loss CSV)",
                          align = (:center, :center), color = c_secondary,
                          fontsize = 14)
            if theme !== nothing
                style_axis!(ax, theme)
            end
        end
    end

    # ---- status bar -------------------------------------------------
    GLMakie.Label(fig[9, 1:4],
        GLMakie.lift(snap_obs) do snap
            snap === nothing && return "No snapshot at this iteration."
            it  = snap["iteration"]
            obj = snap["objective"]
            return @sprintf("Snapshot  iteration %d  |  objective = %.6e", it, obj)
        end,
        fontsize = 11, halign = :left, color = c_secondary)

    # ---- row / column sizing ----------------------------------------
    GLMakie.rowsize!(fig.layout, 1, 36)    # title
    GLMakie.rowsize!(fig.layout, 8, 36)    # interval slider
    GLMakie.rowsize!(fig.layout, 9, 24)    # status

    GLMakie.colgap!(fig.layout, 16)
    GLMakie.rowgap!(fig.layout, 8)

    GLMakie.display(fig)
    return fig
end

end # module NNViewer
