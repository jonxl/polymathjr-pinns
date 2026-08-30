#!/usr/bin/env julia

# Static, checkpoint-only report for the family-trained power-series and
# eigenvalue models. This script never calls train_pinn.

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using CSV
using DataFrames
using JSON
using Plots

include("../../architectures/PINN.jl")
using .PINN

const LF = PINN.loss_functions
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
  Generate static charts from existing dual-representation checkpoints.

  Usage:
    julia --project=. scripts/charts/dual_representation_study.jl [RESULTS_ROOT] [CHART_ROOT] [--ng N] [--map-limit L]

  Defaults:
    RESULTS_ROOT  results/powerseriesvseigenvalue
    CHART_ROOT    charts/dual-representation-study
    --ng          81
    --map-limit   4.0
  """)
end

function parse_args(args)
  positional = String[]
  ng = 81
  map_limit = 4.0f0
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
  results_root = length(positional) >= 1 ? positional[1] : "results/powerseriesvseigenvalue"
  chart_root = length(positional) >= 2 ? positional[2] : "charts/dual-representation-study"
  return (; results_root, chart_root, ng, map_limit)
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

function require_inputs(root)
  missing = String[]
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
    path = joinpath(root, "data", "batch_transfer_$(rep).json")
    isfile(path) || push!(missing, path)
  end
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
    push!(panels, p)
  end
  fig = plot(panels...; layout=(2, 3), size=(1500, 900),
             plot_title="Optimized total loss — representation comparison")
  save_png(fig, joinpath(output_dir, "side-by-side-total-loss.png"))
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

function ordered_plane(limit, ng)
  taus = collect(range(-limit, limit, length=ng))
  deltas = collect(range(-limit, limit, length=ng))
  ts = repeat(Float32.(taus), outer=ng)
  ds = repeat(Float32.(deltas), inner=ng)
  return Float32.(taus), Float32.(deltas), ts, ds
end

function model_inputs(rep, ts, ds)
  if rep === :eigenvalue
    return permutedims(hcat(ts, ds))
  end
  X = zeros(Float32, 3, length(ts))
  for i in eachindex(ts)
    X[:, i] .= LF.canonicalize_alpha(Float32[ds[i], -ts[i], 1.0f0])
  end
  return X
end

function analytic_solutions(ts, ds, xs)
  U = zeros(Float32, length(xs), length(ts))
  for i in eachindex(ts)
    U[:, i] .= LF.eig_true_solution(ts[i], ds[i], 1.0f0, 0.0f0, xs)
  end
  return U
end

function evaluate_error_map(checkpoint, expected_rep, taus, deltas, ts, ds, xs, utrue)
  net, params, state, metadata = PINN.SafeTensorSnapshots.load_any_model(checkpoint)
  actual_rep = Symbol(metadata["representation"])
  actual_rep === expected_rep || error("$checkpoint is $actual_rep, expected $expected_rep")
  out = Array(first(net(model_inputs(expected_rep, ts, ds), params, state)))
  upred = if expected_rep === :power_series
    powers = Float32[x^(n - 1) for x in xs, n in 1:size(out, 1)]
    powers * out
  else
    mu = view(out, 1:1, :); k = view(out, 2:2, :)
    A = view(out, 3:3, :); B = view(out, 4:4, :)
    pterm = LF.PTERM
    xe = Float32[x^(2n) for x in xs, n in 0:pterm]
    xo = Float32[x^(2n + 1) for x in xs, n in 0:pterm]
    kpow = k .^ Float32.(collect(0:pterm))
    C = xe * (LF.CS_COEFF_C .* kpow)
    S = xo * (LF.CS_COEFF_S .* kpow)
    exp.(xs * mu) .* (A .* C .+ B .* S)
  end
  numer = vec(sum(abs2, upred .- utrue; dims=1))
  denom = vec(sum(abs2, utrue; dims=1)) .+ 1f-12
  errors = sqrt.(numer ./ denom)
  return permutedims(reshape(errors, length(taus), length(deltas)))
end

function contour_panel(taus, deltas, errors, title, clims; colorbar=true)
  logerr = log10.(max.(errors, 1e-5))
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

function write_contour_charts(results_root, output_dir, ng, limit)
  taus, deltas, ts, ds = ordered_plane(limit, ng)
  xs = Float32.(collect(range(0.0f0, 1.0f0, length=50)))
  utrue = analytic_solutions(ts, ds, xs)
  maps = Dict{Tuple{Symbol,Symbol},Matrix{Float32}}()
  for rep in REPRESENTATIONS, region in REGIONS
    checkpoint = final_checkpoint(checkpoint_dir(results_root, rep, region))
    println("Evaluating $rep / $region")
    maps[(rep, region)] = evaluate_error_map(checkpoint, rep, taus, deltas, ts, ds, xs, utrue)
  end
  all_log_errors = vcat([vec(log10.(max.(m, 1e-5))) for m in values(maps)]...)
  clims = (max(-5.0, minimum(all_log_errors)), min(2.0, maximum(all_log_errors)))

  for rep in REPRESENTATIONS
    panels = [contour_panel(taus, deltas, maps[(rep, region)], REGION_TITLES[region], clims)
              for region in REGIONS]
    fig = plot(panels...; layout=(2, 3), size=(1500, 950),
               plot_title="Family-trained generalization — $(replace(String(rep), "_" => " ")) | contours ε=10⁻⁴…10⁰")
    save_png(fig, joinpath(output_dir, "$(replace(String(rep), "_" => "-"))-all-families.png"))
  end

  panels = Plots.Plot[]
  for region in REGIONS, rep in REPRESENTATIONS
    push!(panels, contour_panel(taus, deltas, maps[(rep, region)],
      "$(REGION_TITLES[region]) — $(replace(String(rep), "_" => " "))", clims))
  end
  fig = plot(panels...; layout=(6, 2), size=(1500, 3000),
             plot_title="Family-trained generalization — representation comparison | contours ε=10⁻⁴…10⁰")
  save_png(fig, joinpath(output_dir, "side-by-side-contours.png"))
end

function main(args=ARGS)
  opts = parse_args(args)
  opts === nothing && return nothing
  require_inputs(opts.results_root)
  mkpath(opts.chart_root)
  write_loss_charts(opts.results_root, joinpath(opts.chart_root, "training-loss"))
  write_transfer_charts(opts.results_root, joinpath(opts.chart_root, "transfer"))
  write_contour_charts(opts.results_root, joinpath(opts.chart_root, "family-generalization"),
                       opts.ng, opts.map_limit)
  println("Study charts written under $(opts.chart_root)")
  return nothing
end

main()
