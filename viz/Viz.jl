# Visualization and diagnostics: renders solver results with
# Plots.jl into PNG report files under an output directory, and
# dumps the underlying data as CSV tables.
#
# Depends on boltzmann/ (moments, local_fields) being included
# first. One file per group of related graphs:
#
#   convergence.png             — Picard errors, density settling
#   totals_vs_iteration.png     — totals at final time across Picard iterates
#   diagnostics_time.csv        — the collected per-time totals
#   diagnostics_iteration.csv   — the collected per-iterate totals
#
# Everything with a TIME axis renders interactively instead, in
# the GLMakie explorer (viz/Interactive.jl via explore.jl).

using Plots
using Plots.PlotMeasures
using DataFrames
using CSV

ENV["GKSwstype"] = "100"   # headless GR: render offscreen

# Palette: categorical series slots, sequential and diverging
# color gradients.
const SERIES_COLORS = ["#2a78d6" "#eb6834" "#1baf7a"]
const SEQ_GRAD = cgrad([:white, "#2a78d6", "#143a66"])
const DIV_GRAD = cgrad(["#2a78d6", "#f0efec", "#eb6834"])

# ---------------------------------------------------------------
# Data collection: domain-integrated diagnostics
# ---------------------------------------------------------------

# Domain-integrated (nx, ny) moment: one scalar per species.
# (0,0) = total mass; (1,0)/(0,1) = total momentum components.
function total_moment(F, nx::Int, ny::Int, grid::PhaseGrid)
  dA = (grid.x[2] - grid.x[1]) * (grid.y[2] - grid.y[1])
  return [sum(M) * dA for M in moments(F, nx, ny, grid)]
end

# Total kinetic energy per species: ½ ∬ |v|² f.
total_energy(F, grid::PhaseGrid) =
  (total_moment(F, 2, 0, grid) .+ total_moment(F, 0, 2, grid)) ./ 2.0

# One diagnostics row per (state, species): mass, momentum
# components, and kinetic energy.
function diagnostics_frame(states, labels, names, grid)
  rows = DataFrame(label = Float64[], species = String[],
                   mass = Float64[], px = Float64[], py = Float64[],
                   energy = Float64[], entropy = Float64[],
                   mean_T = Float64[], min_f = Float64[])
  for (lab, F) in zip(labels, states)
    m  = total_moment(F, 0, 0, grid)
    px = total_moment(F, 1, 0, grid)
    py = total_moment(F, 0, 1, grid)
    E  = total_energy(F, grid)
    for i in 1:length(names)
      push!(rows, (lab, names[i], m[i], px[i], py[i], E[i],
                   entropy_functional(F[i], grid),
                   mean_temperature(F[i], grid),
                   min_density(F[i])))
    end
  end
  return rows
end

# ---------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------

# One conservation panel: per-species curves plus their sum,
# for one column of the diagnostics frame.
function conservation_panel(df, names, col, xlabel, title, ylabel)
  xs = unique(df.label)
  total = zeros(length(xs))
  series = [(name, df[df.species .== name, col]) for name in names]
  for (_, ys) in series
    total .+= ys
  end
  # Values at pure round-off scale are numerically zero — say so
  # in the title instead of letting noise masquerade as signal.
  maximum(abs.(total)) < 1e-10 && (title *= " (≈ 0, round-off)")
  p = plot(title = title, xlabel = xlabel, ylabel = ylabel,
           legend = :outerright, titlefontsize = 10, guidefontsize = 8,
           legendfontsize = 7)
  for (i, (name, ys)) in enumerate(series)
    plot!(p, xs, ys; label = name, color = SERIES_COLORS[i],
          marker = :circle, markersize = 3, linewidth = 2)
  end
  plot!(p, xs, total; label = "total (sum)", color = SERIES_COLORS[3],
        linestyle = :dash, linewidth = 2)
  return p
end

# 2×2 totals figure: mass, px, py, energy against one clock.
# figure_title states what a flat curve means on that clock.
function conservation_figure(df, names, xlabel, figure_title)
  quantities = [(:mass,   "total mass"),
                (:px,     "total x-momentum"),
                (:py,     "total y-momentum"),
                (:energy, "total kinetic energy")]
  panels = [conservation_panel(df, names, c, xlabel, t, "domain total")
            for (c, t) in quantities]
  return plot(panels...; layout = (2, 2), size = (1100, 750),
              plot_title = figure_title, plot_titlefontsize = 12,
              left_margin = 8mm, bottom_margin = 6mm)
end

# ---------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------

# Renders the full diagnostic report from a windowed solve.
# history/times: the concatenated trajectory over all windows.
# window_errors: per-window Picard error sequences.
# last_result: the last window's PicardResult (its final time is
# the global final time, so iteration diagnostics apply there).
function visualize(history, mix::Mixture{N};
                   times, window_errors, last_result,
                   outdir::String = "results") where {N}
  grid  = mix.grid
  names = [sp.name for sp in mix.species]
  mkpath(outdir)

  nt = length(history)
  ts = collect(times)

  # -- data collection -------------------------------------------
  df_time = diagnostics_frame(history, ts, names, grid)
  CSV.write(joinpath(outdir, "diagnostics_time.csv"), df_time)

  finals  = [it[end] for it in last_result.iterates]
  df_iter = diagnostics_frame(finals, 0.0:(length(finals) - 1), names, grid)
  CSV.write(joinpath(outdir, "diagnostics_iteration.csv"), df_iter)

  # -- convergence.png -------------------------------------------
  errors_all = reduce(vcat, window_errors)
  its = 1:length(errors_all)
  xt = length(its) <= 20 ? (xticks = its,) : NamedTuple()
  p1 = plot(its, errors_all; yscale = :log10, marker = :circle,
            markersize = 3, color = SERIES_COLORS[1], linewidth = 2, legend = false,
            xlabel = "Picard iterate (all windows, concatenated)",
            ylabel = "‖F⁽ⁿ⁾ − F⁽ⁿ⁻¹⁾‖ (L², window)",
            title = "solver contraction per window",
            titlefontsize = 10, xt...)
  # Mark where one time window ends and the next restarts.
  bounds = cumsum(length.(window_errors))[1:end-1]
  isempty(bounds) || vline!(p1, bounds .+ 0.5; color = :gray, linestyle = :dot, linewidth = 1)
  ρ_seq = moment_sequence(last_result, 0, 0, grid)
  p2 = plot(yscale = :log10, xlabel = "Picard iterate n",
            ylabel = "‖ρ⁽ⁿ⁾ − ρ⁽ⁿ⁻¹⁾‖ (L² over space)",
            title = "density settling (last window, final time)",
            titlefontsize = 10, xt...)
  for i in 1:N
    ys = [sqrt(sum((ρ_seq[n][i] .- ρ_seq[n - 1][i]) .^ 2)) for n in 2:length(ρ_seq)]
    plot!(p2, 1:length(ys), ys; label = names[i], color = SERIES_COLORS[i],
          marker = :circle, markersize = 3, linewidth = 2)
  end
  savefig(plot(p1, p2; layout = (1, 2), size = (1100, 450),
               left_margin = 10mm, bottom_margin = 8mm),
          joinpath(outdir, "convergence.png"))

  # -- conservation figures --------------------------------------
  savefig(conservation_figure(df_iter, names, "Picard iterate", "Totals across iterations (final time)"),
          joinpath(outdir, "totals_vs_iteration.png"))

  return outdir
end
