#!/usr/bin/env julia

# Static, data-only report for the portable dual-representation study bundle.
# This script never trains or runs model inference.

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using CSV
using DataFrames
using JSON
using Plots

const REGIONS = [:saddle, :stable_node, :unstable_node,
                 :stable_spiral, :unstable_spiral, :center]
const REPRESENTATIONS = [:power_series, :eigenvalue]
const ERROR_CONTOUR_EXPONENTS = [-4.0, -3.0, -2.0, -1.0, 0.0]
const ERROR_CONTOUR_LABELS = ["ε=10⁻⁴", "ε=10⁻³", "ε=10⁻²", "ε=10⁻¹", "ε=10⁰"]
const ERROR_CONTOUR_COLORS = [:white, :cyan, :lime, :orange, :magenta]
const REGION_TITLES = Dict(
  :saddle => "Saddle",
  :stable_node => "Stable node",
  :unstable_node => "Unstable node",
  :stable_spiral => "Stable spiral",
  :unstable_spiral => "Unstable spiral",
  :center => "Center",
)

function usage()
  println("""
  Generate static charts from a completed portable study bundle.

  Usage:
    julia --project=. scripts/charts/dual_representation_study.jl [RESULTS_ROOT] [CHART_ROOT] [--ng N] [--map-limit L] [--color-scale shared|panel]

  Defaults:
    RESULTS_ROOT  required
    CHART_ROOT    RESULTS_ROOT/charts
    --ng          81
    --map-limit   4.0
    --color-scale shared (use panel to reveal variation within each map)

  --ng and --map-limit must match the saved evaluation grid. They do not
  evaluate new points or increase the resolution of existing results.
  """)
end

function parse_args(args)
  positional = String[]
  ng = 81
  map_limit = 4.0f0
  color_scale = :shared
  i = 1
  while i <= length(args)
    arg = args[i]
    if arg in ("--help", "-h")
      usage()
      return nothing
    elseif arg == "--ng"
      i == length(args) && error("--ng requires an integer")
      i += 1
      ng = parse(Int, args[i])
    elseif arg == "--map-limit"
      i == length(args) && error("--map-limit requires a number")
      i += 1
      map_limit = parse(Float32, args[i])
    elseif arg == "--color-scale"
      i == length(args) && error("--color-scale requires shared or panel")
      i += 1
      color_scale = Symbol(args[i])
      color_scale in (:shared, :panel) || error("--color-scale requires shared or panel")
    elseif startswith(arg, "--")
      error("Unknown option: $arg")
    else
      push!(positional, arg)
    end
    i += 1
  end
  length(positional) <= 2 || error("Expected at most RESULTS_ROOT and CHART_ROOT")
  ng >= 3 || error("--ng must be at least 3")
  map_limit > 0 || error("--map-limit must be positive")
  length(positional) >= 1 || error("RESULTS_ROOT is required")
  results_root = positional[1]
  chart_root = length(positional) >= 2 ? positional[2] : joinpath(results_root, "charts")
  return (; results_root, chart_root, ng, map_limit, color_scale)
end

function checkpoint_dir(root, rep, region)
  current = joinpath(root, "family", String(region), String(rep))
  return isdir(current) ? current : joinpath(root, "exp-batch_transfer-$(rep)-train_$(region)")
end

function final_checkpoint(dir)
  legacy = joinpath(dir, "model.checkpoint")
  isfile(legacy) && return legacy
  candidates = sort(filter(name -> startswith(name, "model-dense-mlp-p") &&
                                   endswith(name, ".checkpoint"), readdir(dir)))
  length(candidates) == 1 || error("Expected exactly one final model checkpoint in $dir, found $(length(candidates))")
  return joinpath(dir, only(candidates))
end

function selected_checkpoint(root, rep, region)
  path = joinpath(root, "selection", "selected-checkpoints.json")
  isfile(path) || error("Missing validation selection: $path")
  selection = JSON.parsefile(path)
  key = "family/$(region)/$(rep)"
  haskey(selection["models"], key) || error("Selection does not contain $key")
  checkpoint = joinpath(root, selection["models"][key]["checkpoint"])
  isfile(checkpoint) || error("Missing selected checkpoint: $checkpoint")
  return checkpoint
end

function require_inputs(root)
  missing = String[]
  selection_path = joinpath(root, "selection", "selected-checkpoints.json")
  isfile(selection_path) || push!(missing, selection_path)
  for rep in REPRESENTATIONS, region in REGIONS
    dir = checkpoint_dir(root, rep, region)
    try
      final_checkpoint(dir)
    catch
      push!(missing, joinpath(dir, "model-dense-mlp-pNNNNNN.checkpoint"))
    end
    isfile(joinpath(dir, "loss.csv")) || push!(missing, joinpath(dir, "loss.csv"))
  end
  for rep in REPRESENTATIONS
    dir = joinpath(root, "global", String(rep))
    try
      final_checkpoint(dir)
    catch
      push!(missing, joinpath(dir, "model-dense-mlp-pNNNNNN.checkpoint"))
    end
    isfile(joinpath(dir, "loss.csv")) || push!(missing, joinpath(dir, "loss.csv"))
  end
  for rep in REPRESENTATIONS
    path = joinpath(root, "data", "batch_transfer_$(rep).json")
    isfile(path) || push!(missing, path)
  end
  final_test = joinpath(root, "data", "final-test-metrics.json")
  isfile(final_test) || push!(missing, final_test)
  suite = joinpath(root, "data", "generalization-suite.json")
  isfile(suite) || push!(missing, suite)
  isempty(missing) || error("Missing study inputs:\n  " * join(missing, "\n  "))
end

function save_png(plot_obj, path)
  mkpath(dirname(path))
  savefig(plot_obj, path)
  println("Wrote $path")
end

positive(v) = max.(Float64.(v), 1e-12)

function loss_panel(csv_path, title)
  df = CSV.read(csv_path, DataFrame)
  p = plot(df.iteration, positive(df.total); label="total (optimized)", lw=2.3,
           color=:black, yscale=:log10, title=title, xlabel="iteration",
           ylabel="loss", legend=:bottomleft, titlefontsize=10,
           legendfontsize=7, guidefontsize=8, tickfontsize=7)
  plot!(p, df.iteration, positive(df.pde); label="PDE", lw=1.7)
  plot!(p, df.iteration, positive(df.supervised); label="supervised", lw=1.7)
  plot!(p, df.iteration, positive(df.bc); label="BC/IC (diagnostic)", lw=1.5, ls=:dash)
  return p
end

function write_loss_charts(results_root, output_dir)
  for rep in REPRESENTATIONS
    panels = [loss_panel(joinpath(checkpoint_dir(results_root, rep, region), "loss.csv"),
                         REGION_TITLES[region]) for region in REGIONS]
    fig = plot(panels...; layout=(2, 3), size=(1500, 900),
               plot_title="Training losses — $(replace(String(rep), "_" => " "))")
    save_png(fig, joinpath(output_dir, "$(replace(String(rep), "_" => "-"))-all-families.png"))
  end

  panels = Plots.Plot[]
  for region in REGIONS
    ps = CSV.read(joinpath(checkpoint_dir(results_root, :power_series, region), "loss.csv"), DataFrame)
    ev = CSV.read(joinpath(checkpoint_dir(results_root, :eigenvalue, region), "loss.csv"), DataFrame)
    p = plot(ps.iteration, positive(ps.total); label="power series", lw=2,
             color=:dodgerblue, yscale=:log10, title=REGION_TITLES[region],
             xlabel="iteration", ylabel="total loss", legend=:bottomleft,
             titlefontsize=10, legendfontsize=7, guidefontsize=8, tickfontsize=7)
    plot!(p, ev.iteration, positive(ev.total); label="eigenvalue", lw=2,
          color=:darkorange, ls=:dash)
    if :power_series_variation in REPRESENTATIONS
      weighted = CSV.read(joinpath(checkpoint_dir(results_root, :power_series_variation, region), "loss.csv"), DataFrame)
      plot!(p, weighted.iteration, positive(weighted.total); label="power series variation", lw=2, color=:purple)
    end
    push!(panels, p)
  end
  fig = plot(panels...; layout=(2, 3), size=(1500, 900),
             plot_title="Optimized total loss — representation comparison")
  save_png(fig, joinpath(output_dir, "side-by-side-total-loss.png"))

  global_panels = [loss_panel(joinpath(results_root, "global", String(rep), "loss.csv"),
                              "Global — $(replace(String(rep), "_" => " "))")
                   for rep in REPRESENTATIONS]
  save_png(plot(global_panels...; layout=(1, length(REPRESENTATIONS)), size=(1400, 600),
                plot_title="Global-model training losses"),
           joinpath(output_dir, "global-side-by-side-loss.png"))
end

function transfer_matrices(path)
  data = JSON.parsefile(path)
  wanted = ["transfer_relerr", "transfer_pde", "transfer_bc", "transfer_sup"]
  by_id = Dict(panel["id"] => panel for panel in data["panels"])
  return Dict(id => reduce(vcat, [permutedims(Float64.(row)) for row in by_id[id]["data"]["matrix"]])
              for id in wanted)
end

function transfer_panel(matrix, title, clims; colorbar=true)
  values = log10.(max.(matrix, 1e-20))
  p = heatmap(1:6, 1:6, values; c=:viridis, clims=clims, yflip=true,
              xticks=(1:6, replace.(String.(REGIONS), "_" => " ")),
              yticks=(1:6, replace.(String.(REGIONS), "_" => " ")),
              xrotation=35, xlabel="tested on", ylabel="trained on", title=title,
              colorbar=colorbar, colorbar_title="log10 value", titlefontsize=9,
              guidefontsize=8, tickfontsize=6)
  return p
end

function write_transfer_charts(results_root, output_dir)
  matrices = Dict(rep => transfer_matrices(joinpath(results_root, "data", "batch_transfer_$(rep).json"))
                  for rep in REPRESENTATIONS)
  specs = [("transfer_relerr", "Solution relative L2"),
           ("transfer_pde", "PDE residual"),
           ("transfer_bc", "BC/IC diagnostic"),
           ("transfer_sup", "Supervised component")]
  ranges = Dict(id => extrema(vcat([vec(log10.(max.(matrices[rep][id], 1e-20)))
                                    for rep in REPRESENTATIONS]...)) for (id, _) in specs)

  for rep in REPRESENTATIONS
    panels = [transfer_panel(matrices[rep][id], title, ranges[id]) for (id, title) in specs]
    fig = plot(panels...; layout=(2, 2), size=(1450, 1150),
               plot_title="Cross-family transfer — $(replace(String(rep), "_" => " "))")
    save_png(fig, joinpath(output_dir, "$(replace(String(rep), "_" => "-"))-transfer.png"))
  end

  panels = Plots.Plot[]
  for (id, metric_title) in specs, rep in REPRESENTATIONS
    push!(panels, transfer_panel(matrices[rep][id],
      "$metric_title — $(replace(String(rep), "_" => " "))", ranges[id]))
  end
  fig = plot(panels...; layout=(4, 2), size=(1500, 2100),
             plot_title="Cross-family transfer — representation comparison")
  save_png(fig, joinpath(output_dir, "side-by-side-transfer.png"))
end

function contour_panel(taus, deltas, errors, title, clims; colorbar=true, color_scale=:shared)
  logerr = log10.(max.(errors, 1e-5))
  lo, hi = extrema(logerr)
  if color_scale === :panel
    clims = lo == hi ? (lo - 0.5, hi + 0.5) : (lo, hi)
    title *= " (local colors)"
  end
  title *= "\nrelative L2: $(round(minimum(errors); sigdigits=3))–$(round(maximum(errors); sigdigits=3))"
  p = heatmap(taus, deltas, logerr; c=:viridis, clims=clims,
              xlabel="τ", ylabel="Δ", title=title, colorbar=colorbar,
              colorbar_title="log10 relative L2", aspect_ratio=:equal,
              xlims=(first(taus), last(taus)), ylims=(first(deltas), last(deltas)),
              titlefontsize=9, guidefontsize=8, tickfontsize=7)
  # One isoline per decade of solution error. Draw levels separately so the
  # legend can use scientific notation rather than raw log10 labels (-4…0).
  for (level, label, color) in zip(ERROR_CONTOUR_EXPONENTS,
                                   ERROR_CONTOUR_LABELS,
                                   ERROR_CONTOUR_COLORS)
    lo < level < hi || continue
    contour!(p, taus, deltas, logerr; levels=[level], color=color, lw=1.8,
             colorbar_entry=false, label=false)
    # Contour series do not reliably create legend entries across backends.
    plot!(p, [NaN], [NaN]; color=color, lw=2, label=label)
  end
  plot!(p; legend=:bottomleft, legendfontsize=5, legend_columns=1)
  plot!(p, taus, zeros(length(taus)); color=:black, ls=:dot, lw=1,
        label=false)
  plot!(p, taus, taus .^ 2 ./ 4; color=:black, ls=:dash, lw=1,
        label=false)
  return p
end

function saved_error_map(entry, ng, limit)
  taus = collect(range(-limit, limit, length=ng))
  deltas = collect(range(-limit, limit, length=ng))
  matrix = fill(Float32(NaN), ng, ng)
  for (point, error) in zip(entry["points"], entry["errors"])
    i = argmin(abs.(taus .- Float32(point["tau"])))
    j = argmin(abs.(deltas .- Float32(point["delta"])))
    matrix[j, i] = Float32(error)
  end
  all(isfinite, matrix) || error("Saved generalization map is incomplete")
  return Float32.(taus), Float32.(deltas), matrix
end

function write_contour_charts(results_root, output_dir, ng, limit; color_scale=:shared)
  suite = JSON.parsefile(joinpath(results_root, "data", "generalization-suite.json"))
  maps = Dict{Tuple{Symbol,Symbol},Matrix{Float32}}()
  axes = nothing
  for rep in REPRESENTATIONS, region in REGIONS
    entry = suite["results"][String(rep)]["generalization_radius"][String(region)]
    taus, deltas, matrix = saved_error_map(entry, ng, limit)
    axes = (taus, deltas)
    maps[(rep, region)] = matrix
  end
  taus, deltas = axes
  all_log_errors = vcat([vec(log10.(max.(m, 1e-5))) for m in values(maps)]...)
  clims = (max(-5.0, minimum(all_log_errors)), min(2.0, maximum(all_log_errors)))

  for rep in REPRESENTATIONS
    panels = [contour_panel(taus, deltas, maps[(rep, region)], REGION_TITLES[region], clims; color_scale)
              for region in REGIONS]
    fig = plot(panels...; layout=(2, 3), size=(1500, 950),
               plot_title="Family-trained generalization — $(replace(String(rep), "_" => " ")) | contours ε=10⁻⁴…10⁰")
    save_png(fig, joinpath(output_dir, "$(replace(String(rep), "_" => "-"))-all-families.png"))
  end

  panels = Plots.Plot[]
  for region in REGIONS, rep in REPRESENTATIONS
    push!(panels, contour_panel(taus, deltas, maps[(rep, region)],
      "$(REGION_TITLES[region]) — $(replace(String(rep), "_" => " "))", clims; color_scale))
  end
  fig = plot(panels...; layout=(6, 2), size=(1500, 3000),
             plot_title="Family-trained generalization — representation comparison | contours ε=10⁻⁴…10⁰")
  save_png(fig, joinpath(output_dir, "side-by-side-contours.png"))

  global_panels = Plots.Plot[]
  for rep in REPRESENTATIONS
    entry = suite["results"][String(rep)]["generalization_radius"]["global"]
    _, _, matrix = saved_error_map(entry, ng, limit)
    push!(global_panels, contour_panel(taus, deltas, matrix,
      "Global — $(replace(String(rep), "_" => " "))", clims; color_scale))
  end
  save_png(plot(global_panels...; layout=(1, length(REPRESENTATIONS)), size=(1400, 650),
                plot_title="Global-model generalization"),
           joinpath(output_dir, "global-side-by-side-contours.png"))
end

function write_extrapolation_charts(results_root, output_dir)
  suite = JSON.parsefile(joinpath(results_root, "data", "generalization-suite.json"))
  for rep in REPRESENTATIONS
    panels = Plots.Plot[]
    data = suite["results"][String(rep)]["extrapolation"]
    for region in REGIONS
      entries = data[String(region)]
      radii = sort(parse.(Int, collect(keys(entries))))
      errors = [entries[string(radius)]["relerr"] for radius in radii]
      push!(panels, plot(radii, positive(errors); marker=:circle, lw=2,
        yscale=:log10, label=false, title=REGION_TITLES[region],
        xlabel="shell radius R", ylabel="relative L2"))
    end
    save_png(plot(panels...; layout=(2, 3), size=(1500, 900),
                  plot_title="In-family extrapolation — $(replace(String(rep), "_" => " "))"),
             joinpath(output_dir, "$(replace(String(rep), "_" => "-"))-extrapolation.png"))
  end
end

function write_range_charts(results_root, output_dir)
  suite = JSON.parsefile(joinpath(results_root, "data", "generalization-suite.json"))
  for rep in REPRESENTATIONS
    panels = Plots.Plot[]
    data = suite["results"][String(rep)]["cross_region_range"]
    for train_region in REGIONS
      p = plot(; yscale=:log10, title="trained: $(replace(String(train_region), "_" => " "))",
               xlabel="shell radius R", ylabel="relative L2", legendfontsize=6)
      entries = data[String(train_region)]
      for test_region in REGIONS
        radii = sort(parse.(Int, [split(key, "/")[2] for key in keys(entries)
                                  if startswith(key, "$(test_region)/")]))
        errors = [entries["$(test_region)/$(radius)"]["relerr"] for radius in radii]
        plot!(p, radii, positive(errors); marker=:circle, lw=1.7,
              label=replace(String(test_region), "_" => " "))
      end
      push!(panels, p)
    end
    save_png(plot(panels...; layout=(2, 3), size=(1600, 950),
                  plot_title="Cross-family range — $(replace(String(rep), "_" => " "))"),
             joinpath(output_dir, "$(replace(String(rep), "_" => "-"))-range.png"))
  end
end

function write_showcase_chart(results_root, output_dir)
  suite = JSON.parsefile(joinpath(results_root, "data", "generalization-suite.json"))
  names = ["memorize", "in-family", "out-range-a", "out-range-b",
           "out-family-c", "out-family-d"]
  ps = [suite["results"]["power_series"]["showcase"][name]["relerr"] for name in names]
  ev = [suite["results"]["eigenvalue"]["showcase"][name]["relerr"] for name in names]
  p = bar(names, hcat(positive(ps), positive(ev)); bar_position=:dodge,
    label=["power series" "eigenvalue"], yscale=:log10, xrotation=25,
    ylabel="solution relative L2", title="Global-model showcase")
  save_png(p, joinpath(output_dir, "representation-comparison.png"))

  xs = Float64.(suite["solution_x"])
  panels = Plots.Plot[]
  for name in names
    ps_entry = suite["results"]["power_series"]["showcase"][name]
    ev_entry = suite["results"]["eigenvalue"]["showcase"][name]
    curve = plot(xs, Float64.(ps_entry["truth"]); color=:black, lw=2.6,
                 label="truth", title=replace(name, "-" => " "),
                 xlabel="x", ylabel="u(x)", titlefontsize=9,
                 legendfontsize=6)
    plot!(curve, xs, Float64.(ps_entry["predicted"]); color=:dodgerblue,
          lw=2, ls=:dash, label="power series")
    plot!(curve, xs, Float64.(ev_entry["predicted"]); color=:darkorange,
          lw=2, ls=:dot, label="eigenvalue")
    push!(panels, curve)
  end
  save_png(plot(panels...; layout=(2, 3), size=(1500, 900),
                plot_title="Global-model showcase solution curves"),
           joinpath(output_dir, "solution-curves.png"))
end

function write_coverage_charts(results_root, output_dir)
  suite = JSON.parsefile(joinpath(results_root, "data", "generalization-suite.json"))
  thresholds = 10.0 .^ (-4:0)
  for rep in REPRESENTATIONS
    p = plot(; xscale=:log10, xlabel="relative-L2 threshold ε",
             ylabel="grid fraction with error ≤ ε", ylims=(0, 1),
             title="Generalization coverage — $(replace(String(rep), "_" => " "))",
             legendfontsize=6)
    maps = suite["results"][String(rep)]["generalization_radius"]
    for region in REGIONS
      coverage = maps[String(region)]["coverage_fraction"]
      values = [coverage[string(threshold)] for threshold in thresholds]
      plot!(p, thresholds, values; marker=:circle, lw=2,
            label=replace(String(region), "_" => " "))
    end
    coverage = maps["global"]["coverage_fraction"]
    values = [coverage[string(threshold)] for threshold in thresholds]
    plot!(p, thresholds, values; marker=:diamond, lw=3, color=:black,
          label="global")
    save_png(p, joinpath(output_dir,
      "$(replace(String(rep), "_" => "-"))-coverage.png"))
  end
end

function main(args=ARGS)
  opts = parse_args(args)
  opts === nothing && return nothing
  empty!(REPRESENTATIONS)
  append!(REPRESENTATIONS, Symbol.(get(JSON.parsefile(joinpath(opts.results_root, "config.json")), "representations", ["power_series", "eigenvalue"])))
  require_inputs(opts.results_root)
  mkpath(opts.chart_root)
  write_loss_charts(opts.results_root, joinpath(opts.chart_root, "training-loss"))
  write_transfer_charts(opts.results_root, joinpath(opts.chart_root, "transfer"))
  write_contour_charts(opts.results_root, joinpath(opts.chart_root, "family-generalization"),
                       opts.ng, opts.map_limit; color_scale=opts.color_scale)
  write_coverage_charts(opts.results_root, joinpath(opts.chart_root, "generalization-coverage"))
  write_extrapolation_charts(opts.results_root, joinpath(opts.chart_root, "extrapolation"))
  write_range_charts(opts.results_root, joinpath(opts.chart_root, "cross-region-range"))
  write_showcase_chart(opts.results_root, joinpath(opts.chart_root, "showcase"))
  println("Study charts written under $(opts.chart_root)")
  return nothing
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
