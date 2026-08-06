# ===================================================================
# Panels — GLMakie renderers, one per PanelSpec archetype.
#
# Each draw_*! takes a Makie GridPosition and a Panel and renders into
# it. Because every renderer has the same shape, the viewer can lay out
# an arbitrary list of panels without knowing what any of them are:
#
#     draw_panel!(fig[r, c], panel, theme)
#
# These are 1:1 ports of the Plots.jl charts in scripts/. Where a Plots
# behaviour has no direct Makie equivalent it is reproduced explicitly
# (e.g. log10 colour scaling is applied to the data, since Makie's
# colorrange works on the values it is given).
# ===================================================================

module Panels

using GLMakie
using Printf

include("PanelSpec.jl")
using .PanelSpec

export draw_panel!, PANEL_KINDS

const PANEL_KINDS = (:heatmap, :grouped_bar, :lines, :scatter, :grid)

# Default categorical palette. Series without an explicit colour cycle
# through this, so charts stay distinguishable without per-call fuss.
const SERIES_PALETTE = ["#4fc3f7", "#ff8a65", "#66bb6a", "#ef5350",
                        "#ba68c8", "#ffb74d", "#4db6ac", "#f06292"]

series_color(opts, i) = begin
  cs = get(opts, "colors", nothing)
  (cs === nothing || isempty(cs)) ? SERIES_PALETTE[mod1(i, length(SERIES_PALETTE))] :
                                    cs[mod1(i, length(cs))]
end

function series_linestyle(opts, i)
  styles = get(opts, "linestyles", nothing)
  (styles === nothing || isempty(styles)) && return nothing
  s = styles[mod1(i, length(styles))]
  s in ("dash", "dashed") && return :dash
  s == "dot" && return :dot
  s in ("dashdot", "dash_dot") && return :dashdot
  return nothing
end

# ---- theming -------------------------------------------------------
# Mirrors NNViewer.style_axis! so panels match the rest of the window.
# `theme === nothing` leaves Makie defaults alone, which keeps these
# renderers usable standalone (tests, one-off figures).

function style_axis!(ax, theme)
  theme === nothing && return ax
  t = GLMakie.to_color
  ax.backgroundcolor = t(theme.bg_secondary)
  ax.xgridcolor = t(theme.grid_color)
  ax.ygridcolor = t(theme.grid_color)
  ax.xlabelcolor = t(theme.text_primary)
  ax.ylabelcolor = t(theme.text_primary)
  ax.xtickcolor = t(theme.text_primary)
  ax.ytickcolor = t(theme.text_primary)
  ax.xticklabelcolor = t(theme.text_primary)
  ax.yticklabelcolor = t(theme.text_primary)
  ax.titlecolor = t(theme.text_primary)
  ax.xgridvisible = true
  ax.ygridvisible = true
  ax.xgridwidth = 0.5
  ax.ygridwidth = 0.5
  return ax
end

# ---- dispatch ------------------------------------------------------

"""
    draw_panel!(pos, panel::Panel; theme = nothing)

Render `panel` into the Makie `GridPosition` `pos`. Dispatches on
`panel.kind`; unknown kinds raise rather than silently drawing nothing,
so a typo in a spec fails loudly.
"""
function draw_panel!(pos, panel::Panel; theme = nothing)
  if panel.kind === :heatmap
    return draw_heatmap!(pos, panel; theme = theme)
  elseif panel.kind === :grouped_bar
    return draw_grouped_bar!(pos, panel; theme = theme)
  elseif panel.kind === :lines
    return draw_lines!(pos, panel; theme = theme)
  elseif panel.kind === :scatter
    return draw_scatter!(pos, panel; theme = theme)
  elseif panel.kind === :grid
    return draw_grid!(pos, panel; theme = theme)
  else
    error("unknown panel kind :$(panel.kind); expected one of $(PANEL_KINDS)")
  end
end

# ---- :heatmap ------------------------------------------------------
#
# Port of report_matrix: rows = trained on, cols = tested on, colour by
# log10(max(v, floor)), first row at top, numeric text in every cell.

function draw_heatmap!(pos, panel::Panel; theme = nothing)
  rows = panel.data["matrix"]
  row_labels = panel.data["row_labels"]
  col_labels = panel.data["col_labels"]
  nr, nc = length(rows), length(rows[1])

  # rows-of-rows -> (col, row) matrix, which is Makie's orientation
  raw = [Float64(rows[i][j]) for j in 1:nc, i in 1:nr]

  log_color = get(panel.opts, "log_color", true)
  floor_val = Float64(get(panel.opts, "floor_val", 1e-20))
  vals = log_color ? log10.(max.(raw, floor_val)) : raw

  yflip = get(panel.opts, "yflip", true)
  xrot = Float64(get(panel.opts, "xrotation", 30)) * pi / 180
  xvals = get(panel.data, "x_values", nothing)
  yvals = get(panel.data, "y_values", nothing)
  use_coords = xvals !== nothing && yvals !== nothing
  xs = use_coords ? Float64.(xvals) : collect(1:nc)
  ys = use_coords ? Float64.(yvals) : collect(1:nr)

  ax = Axis(pos;
    title = panel.title,
    xlabel = panel.xlabel, ylabel = panel.ylabel,
    xticks = use_coords ? WilkinsonTicks(5) : (1:nc, col_labels),
    yticks = use_coords ? WilkinsonTicks(5) : (1:nr, row_labels),
    xticklabelrotation = xrot,
    yreversed = use_coords ? false : yflip,
  )
  style_axis!(ax, theme)
  ax.xgridvisible = false
  ax.ygridvisible = false

  hm = heatmap!(ax, xs, ys, vals; colormap = :viridis)
  Colorbar(pos[1, 2], hm; label = String(get(panel.opts, "colorbar_label", "log10 value")))

  levels = get(panel.opts, "contour_levels", nothing)
  if levels !== nothing && !isempty(levels)
    contour_vals = log_color ? log10.(max.(Float64.(levels), floor_val)) : Float64.(levels)
    contour!(ax, xs, ys, vals; levels = contour_vals, color = :white, linewidth = 2)
  end

  if get(panel.opts, "annotate", true) && !use_coords
    pct = get(panel.opts, "percent", false)
    for i in 1:nr, j in 1:nc
      v = raw[j, i]
      txt = pct ? string(round(v * 100; sigdigits = 2), "%") :
                  string(round(v; sigdigits = 2))
      text!(ax, j, i; text = txt, align = (:center, :center),
            color = :white, fontsize = 10)
    end
  end
  return ax
end

# ---- :grouped_bar --------------------------------------------------
#
# Port of the component-comparison bars: one group per category, bars
# offset within the group, log10 y, optional per-bar annotation above.

function draw_grouped_bar!(pos, panel::Panel; theme = nothing)
  cats = panel.data["categories"]
  series = panel.data["series"]
  ncat, nser = length(cats), length(series)
  log_y = get(panel.opts, "log_y", true)

  ax = Axis(pos;
    title = panel.title,
    xlabel = panel.xlabel, ylabel = panel.ylabel,
    xticks = (1:ncat, cats),
    yscale = log_y ? log10 : identity,
  )
  style_axis!(ax, theme)

  width = 0.8 / nser
  for (s, ser) in enumerate(series)
    # centre the group on the category tick
    offset = (s - (nser + 1) / 2) * width
    xs = collect(1:ncat) .+ offset
    ys = Float64.(ser["values"])
    # log scale cannot show non-positive values; floor them so the bar
    # is visibly tiny rather than silently dropped.
    if log_y
      pos_vals = filter(>(0), ys)
      floorv = isempty(pos_vals) ? 1e-20 : minimum(pos_vals) * 1e-3
      ys = [v > 0 ? v : floorv for v in ys]
    end
    barplot!(ax, xs, ys; width = width, color = series_color(panel.opts, s),
             label = ser["name"])
  end

  anns = get(panel.data, "annotations", nothing)
  if anns !== nothing
    for (s, ser) in enumerate(series)
      s > length(anns) && continue
      offset = (s - (nser + 1) / 2) * width
      ys = Float64.(ser["values"])
      for c in 1:min(ncat, length(anns[s]))
        isempty(anns[s][c]) && continue
        y = ys[c] > 0 ? ys[c] : 1e-20
        text!(ax, c + offset, y * (log_y ? 2.0 : 1.05);
              text = anns[s][c], align = (:center, :bottom), fontsize = 9,
              color = theme === nothing ? :black : GLMakie.to_color(theme.text_primary))
      end
    end
  end

  nser > 1 && axislegend(ax; position = :rt, framevisible = false)
  return ax
end

# ---- :lines --------------------------------------------------------

function draw_lines!(pos, panel::Panel; theme = nothing)
  series = panel.data["series"]
  log_y = get(panel.opts, "log_y", false)
  log_x = get(panel.opts, "log_x", false)

  ax = Axis(pos;
    title = panel.title, xlabel = panel.xlabel, ylabel = panel.ylabel,
    yscale = log_y ? log10 : identity,
    xscale = log_x ? log10 : identity,
  )
  style_axis!(ax, theme)

  for (s, ser) in enumerate(series)
    xs = Float64.(ser["x"])
    ys = Float64.(ser["y"])
    if log_y   # drop non-positive points; log10 cannot render them
      keep = ys .> 0
      xs, ys = xs[keep], ys[keep]
      isempty(ys) && continue
    end
    linestyle = series_linestyle(panel.opts, s)
    if linestyle === nothing
      lines!(ax, xs, ys; color = series_color(panel.opts, s),
             linewidth = 2, label = ser["name"])
    else
      lines!(ax, xs, ys; color = series_color(panel.opts, s),
             linewidth = 2, linestyle = linestyle, label = ser["name"])
    end
  end

  length(series) > 1 && axislegend(ax; position = :rt, framevisible = false)
  return ax
end

# ---- :scatter ------------------------------------------------------

function draw_scatter!(pos, panel::Panel; theme = nothing)
  series = panel.data["series"]
  ms = Float64(get(panel.opts, "markersize", 6))

  ax = Axis(pos; title = panel.title, xlabel = panel.xlabel, ylabel = panel.ylabel)
  style_axis!(ax, theme)

  for (s, ser) in enumerate(series)
    scatter!(ax, Float64.(ser["x"]), Float64.(ser["y"]);
             color = series_color(panel.opts, s), markersize = ms,
             label = ser["name"])
  end

  length(series) > 1 && axislegend(ax; position = :rt, framevisible = false)
  return ax
end

# ---- :grid ---------------------------------------------------------
#
# Port of plot(panels...; layout = (r, c)). Children are laid out
# row-major into a nested GridLayout, so grids compose arbitrarily.

function draw_grid!(pos, panel::Panel; theme = nothing)
  rows = Int(panel.opts["rows"])
  cols = Int(panel.opts["cols"])
  children = [PanelSpec.from_dict(c) for c in panel.data["children"]]

  gl = GridLayout(pos)
  if !isempty(panel.title)
    Label(gl[0, 1:cols], panel.title; fontsize = 16,
          color = theme === nothing ? :black : GLMakie.to_color(theme.text_primary))
  end
  for (i, child) in enumerate(children)
    r = div(i - 1, cols) + 1
    c = mod(i - 1, cols) + 1
    (r > rows) && break
    draw_panel!(gl[r, c], child; theme = theme)
  end
  return gl
end

end # module Panels
