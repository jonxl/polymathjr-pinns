#!/usr/bin/env julia
# Evaluate existing final checkpoints directly, without training or selecting models.
include("dual_representation_study.jl")
include("../../utils/safetensors_utils.jl")
include("../../utils/generalization_geometry.jl")
using .SafeTensorSnapshots
using .GeneralizationGeometry: exact_solution
using LinearAlgebra, SHA, Statistics

function checkpoint_errors(path, taus, deltas; nx=200, batch_size=1024)
  net, params, state, metadata = SafeTensorSnapshots.load_checkpoint(path)
  rep = Symbol(metadata["representation"])
  width = size(params.layer_1.weight, 2)
  encoding = get(metadata, "input_encoding", width == 2 ? "trace_determinant" : "coefficients")
  encoding in ("trace_determinant", "coefficients") || error("Unknown input encoding: $encoding")
  width == (encoding == "trace_determinant" ? 2 : 3) || error("Unsupported input width: $width")
  points = [(t, d) for d in deltas for t in taus]
  xs = collect(range(0., 1., length=nx))
  errors = Float64[]
  for start in 1:batch_size:length(points)
    chunk = points[start:min(start + batch_size - 1, length(points))]
    inputs = reduce(hcat, [encoding == "trace_determinant" ?
      Float32[t, d] : Float32[d, -t, 1] for (t, d) in chunk])
    output = Float64.(first(net(inputs, params, state)))
    predicted = if rep in (:power_series, :power_series_variation)
      # Raw derivative coefficients: factorials belong in reconstruction.
      basis = ones(Float64, nx, size(output, 1))
      for n in 1:size(output, 1)-1
        basis[:, n+1] .= basis[:, n] .* xs ./ n
      end
      basis * output
    elseif rep === :eigenvalue
      [exact_solution(2output[1,j], output[1,j]^2-output[2,j], x;
        A=output[3,j], derivative=output[1,j]*output[3,j]+output[4,j])
        for x in xs, j in axes(output, 2)]
    else
      error("Unsupported representation: $rep")
    end
    truth = [exact_solution(t, d, x) for x in xs, (t, d) in chunk]
    batch_errors = vec(sqrt.(sum(abs2, predicted .- truth; dims=1) ./
                            (sum(abs2, truth; dims=1) .+ 1e-12)))
    append!(errors, [isfinite(e) ? e : Inf for e in batch_errors])
  end
  # Plotting and CSV convention: rows are determinant, columns are trace.
  matrix = permutedims(reshape(errors, length(taus), length(deltas)))
  return matrix, metadata
end

function error_stats(matrix)
  es = vec(matrix)
  finite = filter(isfinite, es)
  Dict("points"=>length(es), "failed_points"=>count(!isfinite, es),
       "minimum"=>isempty(finite) ? nothing : minimum(finite),
       "median_finite"=>isempty(finite) ? nothing : median(finite),
       "maximum_finite"=>isempty(finite) ? nothing : maximum(finite),
       "fraction_below_0.1"=>count(e->e<=.1, es)/length(es),
       "fraction_below_0.01"=>count(e->e<=.01, es)/length(es))
end

function regenerate_maps(args=ARGS)
  if length(args) < 2 || length(args) > 4
    error("Usage: julia --project=. scripts/charts/checkpoint_error_maps.jl RESULTS_ROOT OUTPUT_ROOT [NG=161] [LIMIT=4]")
  end
  root, outdir = args[1:2]
  ng = length(args) >= 3 ? parse(Int, args[3]) : 161
  limit = length(args) >= 4 ? parse(Float64, args[4]) : 4.
  ng >= 3 && isodd(ng) || error("NG must be odd and at least 3")
  isfinite(limit) && limit > 0 || error("LIMIT must be finite and positive")
  BLAS.set_num_threads(2)
  axis = Float32.(range(-limit, limit, length=ng))
  maps = Dict{Tuple{Symbol,Symbol},Matrix{Float64}}()
  summary = Dict{String,Any}("grid_size"=>ng, "map_limit"=>limit,
    "solution_interval"=>[0.,1.], "solution_points"=>200,
    "initial_conditions"=>[1.,0.], "power_series_output_basis"=>"taylor_derivative",
    "checkpoint_policy"=>"saved final checkpoints; no validation selection",
    "models"=>Dict{String,Any}())
  mkpath(outdir)
  regions = copy(REGIONS)
  if all(isdir(joinpath(root,"global",String(rep))) for rep in REPRESENTATIONS)
    push!(regions, :global)
  end
  for region in regions, rep in REPRESENTATIONS
    dir = region === :global ? joinpath(root,"global",String(rep)) : checkpoint_dir(root,rep,region)
    path = final_checkpoint(dir)
    println("Evaluating $region / $rep on $(ng)×$(ng) grid")
    matrix, metadata = checkpoint_errors(path, axis, axis)
    maps[(rep,region)] = matrix
    key = "$(region)/$(rep)"
    summary["models"][key] = Dict("checkpoint"=>abspath(path),
      "sha256"=>open(sha256,path) |> bytes2hex,
      "input_encoding"=>get(metadata,"input_encoding","inferred from input width"),
      "fine_grid"=>error_stats(matrix),
      "coarse_grid"=>error_stats(matrix[1:2:end,1:2:end]))
    CSV.write(joinpath(outdir,"$(region)-$(rep)-errors.csv"), DataFrame(
      tau=[t for d in axis for t in axis], delta=[d for d in axis for t in axis],
      relative_l2=vec(permutedims(matrix))))
  end
  open(joinpath(outdir,"summary.json"),"w") do io
    JSON.print(io,summary,2)
  end
  finite_logs = [log10(max(e,1e-5)) for matrix in values(maps) for e in matrix if isfinite(e)]
  isempty(finite_logs) && error("All predictions failed; see summary.json")
  lo, hi = extrema(finite_logs)
  clims = lo == hi ? (lo-.5,hi+.5) : (lo,hi)
  for scale in (:shared,:panel)
    function panel(rep, region)
      matrix = maps[(rep,region)]
      # Mark failed predictions explicitly; cap their display at the maximum color.
      failures = count(!isfinite,matrix)
      display = map(e->isfinite(e) ? e : 10.0^clims[2], matrix)
      title = region === :global ? "Global" : REGION_TITLES[region]
      title *= " — $(replace(String(rep), '_' => ' '))"
      failures > 0 && (title *= " ($failures failed)")
      contour_panel(axis,axis,display,title,clims;color_scale=scale)
    end
    for rep in REPRESENTATIONS
      fig = plot([panel(rep,region) for region in REGIONS]...;
        layout=(2,3),size=(1800,1100),
        plot_title="$(ng)×$(ng) evaluation | $(scale) color scales | Taylor reconstruction")
      save_png(fig,joinpath(outdir,"$(rep)-$(scale).png"))
    end
    comparison = plot([panel(rep,region) for region in REGIONS for rep in REPRESENTATIONS]...;
      layout=(6,2),size=(1600,3200),plot_title="$(ng)×$(ng) evaluation | $(scale) color scales")
    save_png(comparison,joinpath(outdir,"comparison-$(scale).png"))
    if :global in regions
      fig = plot([panel(rep,:global) for rep in REPRESENTATIONS]...;
        layout=(1,2),size=(1500,700),plot_title="Global models | $(ng)×$(ng) | $(scale) colors")
      save_png(fig,joinpath(outdir,"global-$(scale).png"))
    end
  end
  println("Saved maps and resolution comparison to $outdir")
end

abspath(PROGRAM_FILE) == (@__FILE__) && regenerate_maps()
