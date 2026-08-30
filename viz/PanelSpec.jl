# ===================================================================
# PanelSpec — the contract between experiments and the viewer.
#
# Experiments (scripts/, training runs) EMIT specs; the viewer DRAWS
# them. Neither knows about the other. Adding a new chart means
# emitting a spec, never editing NNViewer.
#
# Every chart in the project reduces to one of five archetypes:
#
#   :heatmap      matrix + row/col labels, optional log10 colour,
#                 optional per-cell text  (cross-region transfer
#                 matrices, generalization-radius maps)
#   :grouped_bar  N series of bars over shared categories, optional
#                 log y, optional per-bar annotations  (loss-component
#                 in-family vs out-of-family comparisons)
#   :lines        N named series over a shared or per-series x
#                 (solutions, errors, loss histories)
#   :scatter      N named point clouds  ((τ,Δ) region plots)
#   :grid         a composite of child panels laid out (rows, cols)
#
# A spec is plain data, so it serializes to JSON and can be produced by
# a Julia script, replayed from disk, or built on the fly in-memory.
# ===================================================================

module PanelSpec

using JSON

export Panel, PanelSet, to_dict, from_dict, save_panels, load_panels,
       heatmap_panel, grouped_bar_panel, lines_panel, scatter_panel, grid_panel

"""
    Panel

One chart. `kind` selects the archetype and therefore which `draw_*!`
renders it; `data` holds the archetype-specific payload; `opts` holds
presentation flags that are safe to ignore if a backend can't honour them.

Fields
- `kind`   — `:heatmap`, `:grouped_bar`, `:lines`, `:scatter`, `:grid`
- `id`     — stable identifier, used for registry lookup and layout order
- `title`  — chart title
- `xlabel` / `ylabel` — axis labels ("" to omit)
- `data`   — archetype payload (see the constructors below)
- `opts`   — e.g. `:log_color`, `:log_y`, `:yflip`, `:annotate`, `:xrotation`
"""
struct Panel
  kind::Symbol
  id::String
  title::String
  xlabel::String
  ylabel::String
  data::Dict{String,Any}
  opts::Dict{String,Any}
end

"""
    PanelSet

A named collection of panels — what the viewer's dropdown selects between.
`meta` carries provenance (representation, run id, source script) so the
viewer can label the view without re-deriving anything.
"""
struct PanelSet
  name::String
  meta::Dict{String,Any}
  panels::Vector{Panel}
end

# ---------------------------------------------------------------------------
# Archetype constructors — these define the payload shape for each kind.
# ---------------------------------------------------------------------------

"""
    heatmap_panel(id, title, matrix, row_labels, col_labels; kwargs...)

Matrix chart. Mirrors `report_matrix` from the experiment scripts:
log10 colour over a floored value, rows = trained-on, cols = tested-on,
per-cell numeric annotation.

- `log_color`  — colour by log10(max(v, floor)) instead of raw value
- `floor_val`  — the floor applied before log10 (guards log10(0))
- `annotate`   — draw the numeric value in each cell
- `percent`    — format annotations as percentages
- `yflip`      — first row at the top (matches the scripts' orientation)
- `style`      — `"cells"` (discrete heatmap, default) or `"filled_contour"`
                 (banded contourf, for continuous fields like error maps)
- `interpolate`— bilinearly smooth the `"cells"` rendering
- `contour_color`, `contour_width` — styling for the ε-contour overlay
- `clims`      — fixed `(lo, hi)` colour limits, so sibling panels share a scale
"""
function heatmap_panel(id, title, matrix, row_labels, col_labels;
                       xlabel = "", ylabel = "", log_color = true,
                       floor_val = 1e-20, annotate = true, percent = false,
                       yflip = true, colorbar_label = "log10 value", xrotation = 30,
                       x_values = nothing, y_values = nothing, contour_levels = nothing,
                       style = "cells", interpolate = false,
                       contour_color = "white", contour_width = 2, clims = nothing)
  return Panel(:heatmap, id, title, xlabel, ylabel,
    Dict{String,Any}(
      "matrix" => [collect(Float64.(matrix[i, :])) for i in 1:size(matrix, 1)],
      "row_labels" => String.(collect(row_labels)),
      "col_labels" => String.(collect(col_labels)),
      "x_values" => x_values === nothing ? nothing : collect(Float64.(x_values)),
      "y_values" => y_values === nothing ? nothing : collect(Float64.(y_values)),
    ),
    Dict{String,Any}(
      "log_color" => log_color, "floor_val" => floor_val,
      "annotate" => annotate, "percent" => percent, "yflip" => yflip,
      "colorbar_label" => colorbar_label, "xrotation" => xrotation,
      "contour_levels" => contour_levels === nothing ? nothing : collect(Float64.(contour_levels)),
      "style" => String(style), "interpolate" => interpolate,
      "contour_color" => String(contour_color), "contour_width" => contour_width,
      "clims" => clims === nothing ? nothing : collect(Float64.(clims)),
    ))
end

"""
    grouped_bar_panel(id, title, categories, series; kwargs...)

Grouped bars. `series` is a vector of `(name, values)` pairs, one bar per
category per series. `annotations` is an optional vector of vectors of
strings, parallel to `series`, drawn above the corresponding bars —
used by the scripts for the "×N" out/in ratio labels.
"""
function grouped_bar_panel(id, title, categories, series;
                           xlabel = "", ylabel = "", log_y = true,
                           annotations = nothing, colors = nothing)
  return Panel(:grouped_bar, id, title, xlabel, ylabel,
    Dict{String,Any}(
      "categories" => String.(collect(categories)),
      "series" => [Dict{String,Any}("name" => String(n),
                                    "values" => collect(Float64.(v))) for (n, v) in series],
      "annotations" => annotations === nothing ? nothing :
                       [String.(collect(a)) for a in annotations],
    ),
    Dict{String,Any}("log_y" => log_y,
                     "colors" => colors === nothing ? nothing : String.(collect(colors))))
end

"""
    lines_panel(id, title, series; kwargs...)

Multi-series line chart. `series` is a vector of `(name, xs, ys)` tuples,
so series may have different x grids (loss histories of differing length,
analytic curve on a fine grid vs prediction on collocation points).
"""
function lines_panel(id, title, series;
                     xlabel = "", ylabel = "", log_y = false, log_x = false,
                     colors = nothing, markers = nothing, linestyles = nothing)
  return Panel(:lines, id, title, xlabel, ylabel,
    Dict{String,Any}(
      "series" => [Dict{String,Any}("name" => String(n),
                                    "x" => collect(Float64.(xv)),
                                    "y" => collect(Float64.(yv))) for (n, xv, yv) in series],
    ),
    Dict{String,Any}("log_y" => log_y, "log_x" => log_x,
                     "colors" => colors === nothing ? nothing : String.(collect(colors)),
                     "markers" => markers === nothing ? nothing : String.(collect(markers)),
                     "linestyles" => linestyles === nothing ? nothing : String.(collect(linestyles))))
end

"""
    scatter_panel(id, title, series; kwargs...)

Multi-series point cloud — e.g. (τ, Δ) samples coloured by region.
`series` is a vector of `(name, xs, ys)` tuples.
"""
function scatter_panel(id, title, series;
                       xlabel = "", ylabel = "", colors = nothing, markersize = 6)
  return Panel(:scatter, id, title, xlabel, ylabel,
    Dict{String,Any}(
      "series" => [Dict{String,Any}("name" => String(n),
                                    "x" => collect(Float64.(xv)),
                                    "y" => collect(Float64.(yv))) for (n, xv, yv) in series],
    ),
    Dict{String,Any}("colors" => colors === nothing ? nothing : String.(collect(colors)),
                     "markersize" => markersize))
end

"""
    grid_panel(id, title, children; rows, cols)

Composite panel — the scripts' `plot(panels...; layout = (2, 2))`.
Children are themselves `Panel`s, so grids nest.
"""
function grid_panel(id, title, children; rows::Int, cols::Int)
  return Panel(:grid, id, title, "", "",
    Dict{String,Any}("children" => [to_dict(c) for c in children]),
    Dict{String,Any}("rows" => rows, "cols" => cols))
end

# ---------------------------------------------------------------------------
# Serialization — specs are plain data, so JSON round-trips exactly.
# ---------------------------------------------------------------------------

to_dict(p::Panel) = Dict{String,Any}(
  "kind" => String(p.kind), "id" => p.id, "title" => p.title,
  "xlabel" => p.xlabel, "ylabel" => p.ylabel, "data" => p.data, "opts" => p.opts,
)

from_dict(d::AbstractDict) = Panel(
  Symbol(d["kind"]), d["id"], d["title"],
  get(d, "xlabel", ""), get(d, "ylabel", ""),
  Dict{String,Any}(d["data"]), Dict{String,Any}(get(d, "opts", Dict())),
)

to_dict(ps::PanelSet) = Dict{String,Any}(
  "name" => ps.name, "meta" => ps.meta, "panels" => [to_dict(p) for p in ps.panels],
)

function panelset_from_dict(d::AbstractDict)
  return PanelSet(d["name"], Dict{String,Any}(get(d, "meta", Dict())),
                  [from_dict(p) for p in d["panels"]])
end

"""
    save_panels(path, panelset)

Write a `PanelSet` to JSON. Experiments call this instead of `savefig`.
"""
function save_panels(path::String, ps::PanelSet)
  mkpath(dirname(path))
  open(path, "w") do io
    JSON.print(io, to_dict(ps))
  end
  return path
end

"""
    load_panels(path) → PanelSet

Read a `PanelSet` written by `save_panels`.
"""
load_panels(path::String) = panelset_from_dict(JSON.parsefile(path))

end # module PanelSpec
