module DualStudyEvaluation

using CSV
using CUDA
using DataFrames
using JSON
using Random
using SHA
using Serialization

include("../architectures/PINN.jl")
using .PINN
include("dual_dataset.jl")
using .DualDataset
include("plugboard.jl")
using .Plugboard

const LF = PINN.loss_functions
const REGIONS = [:saddle, :stable_node, :unstable_node,
                 :stable_spiral, :unstable_spiral, :center]
representations(config) = Symbol.(get(config, "representations", ["power_series", "eigenvalue"]))

export select_checkpoints, run_final_test, run_generalization_suite,
       run_custom_inference, load_study

function _sha256_file(path::AbstractString)
  open(path, "r") do io
    return bytes2hex(sha256(io))
  end
end

function load_study(root::AbstractString; verify_datasets::Bool=true)
  config_path = joinpath(root, "config.json")
  isfile(config_path) || error("Missing study configuration: $config_path")
  config = JSON.parsefile(config_path)
  splits = Dict(
    :train => load_materialized_split(joinpath(root, "datasets", "train"); verify=verify_datasets),
    :validation => load_materialized_split(joinpath(root, "datasets", "validation"); verify=verify_datasets),
    :test => load_materialized_split(joinpath(root, "datasets", "test"); verify=verify_datasets),
  )
  return (; root=String(root), config, splits)
end

function _model_dir(root, scope::Symbol, representation::Symbol, region)
  scope === :global && return joinpath(root, "global", String(representation))
  return joinpath(root, "family", String(region), String(representation))
end

function _model_key(scope::Symbol, representation::Symbol, region)
  scope === :global && return "global/$(representation)"
  return "family/$(region)/$(representation)"
end

function _candidate_metadata(path::AbstractString)
  payload = deserialize(path)
  return Dict{String,Any}(String(k) => v for (k, v) in payload if k != "flat")
end

function _candidate_checkpoints(dir::AbstractString, checkpoint_every::Int, epochs::Int)
  paths = String[]
  snapshot_dir = joinpath(dir, "snapshots")
  isdir(snapshot_dir) || error("Missing snapshot directory: $snapshot_dir")
  append!(paths, [joinpath(snapshot_dir, name) for name in readdir(snapshot_dir)
                  if endswith(name, ".checkpoint")])
  finals = [joinpath(dir, name) for name in readdir(dir)
            if startswith(name, "model-") && endswith(name, ".checkpoint")]
  length(finals) == 1 || error("Expected exactly one final checkpoint in $dir")
  append!(paths, finals)

  by_epoch = Dict{Int,String}()
  for path in paths
    metadata = _candidate_metadata(path)
    epoch = Int(metadata["epoch"])
    haskey(by_epoch, epoch) && error(
      "Duplicate epoch $epoch checkpoints in $dir; this is a mixed or contaminated run")
    by_epoch[epoch] = path
  end
  expected = collect(checkpoint_every:checkpoint_every:epochs)
  sort(collect(keys(by_epoch))) == expected || error(
    "Incomplete checkpoint series in $dir: found $(sort(collect(keys(by_epoch)))), expected $expected")
  return [by_epoch[epoch] for epoch in expected]
end

function _dummy_dataset(split::AbstractODESplit, region)
  index = region === nothing ? 1 : findfirst(==(region), split.regions)
  return Dict(example_at(split, index))
end

function _settings(representation::Symbol, split::AbstractODESplit, region, config)
  training = config["training"]
  N = Int(training["power_series_degree"])
  points = Int(training["collocation_points"])
  width = Int(training["network_hidden_width"])
  seed = Int(training["model_seed"])
  return PINNSettings(
    width, seed, _dummy_dataset(split, region), 1, N, N + 1, points,
    0f0, 1f0, 1f0, 1f0, Float32.(range(0f0, 1f0, length=points)),
    "adam", representation, :trace_determinant,
  )
end

function _settings(representation::Symbol, dataset::Dict, config)
  training = config["training"]
  N = Int(training["power_series_degree"])
  points = Int(training["collocation_points"])
  width = Int(training["network_hidden_width"])
  seed = Int(training["model_seed"])
  return PINNSettings(
    width, seed, dataset, 1, N, N + 1, points,
    0f0, 1f0, 1f0, 1f0, Float32.(range(0f0, 1f0, length=points)),
    "adam", representation, :trace_determinant,
  )
end

_host_scalar(x::Number) = Float64(x)
_host_scalar(x) = Float64(only(Array(x)))

_metrics_are_finite(metrics) = all(isfinite,
  (metrics.relerr, metrics.pde, metrics.bc, metrics.supervised))

function _assert_finite_metrics(metrics, context)
  _metrics_are_finite(metrics) || error(
    "Non-finite inference metrics for $context: " *
    "relerr=$(metrics.relerr), pde=$(metrics.pde), bc=$(metrics.bc), supervised=$(metrics.supervised)")
  return metrics
end

function evaluate_checkpoint(path::AbstractString, representation::Symbol,
                             split::AbstractODESplit, config;
                             region::Union{Symbol,Nothing}=nothing,
                             batch_size::Int=16_384, use_gpu::Bool=CUDA.functional())
  net, params, state, metadata = PINN.SafeTensorSnapshots.load_any_model(path)
  Symbol(metadata["representation"]) === representation ||
    error("Representation mismatch in $path")
  settings = _settings(representation, split, region, config)
  params_device = use_gpu ? CUDA.cu(params) : params
  state_device = use_gpu ? CUDA.cu(state) : state
  to_device = use_gpu ? CUDA.cu : identity
  count = region === nothing ? split.size : family_size(split, region)
  nbatches = cld(count, batch_size)

  numerator = 0.0
  denominator = 0.0
  pde_total = 0.0
  bc_total = 0.0
  sup_total = 0.0
  seen = 0
  for batch in 1:nbatches
    items = evaluation_batch_items(split, batch, batch_size; region=region)
    buffers = representation === :eigenvalue ?
      LF.precompute_batch_eig_buffers(settings, items, use_gpu, to_device) :
      LF.precompute_batch_buffers(settings, items, use_gpu, to_device)
    output = first(net(buffers.X, params_device, state_device))
    pde, bc, sup = representation === :eigenvalue ?
      LF.batched_eigenvalue_losses(output, buffers) :
      LF.batched_power_series_losses(output, buffers; variation=representation === :power_series_variation)
    predicted = LF.batched_reconstruct(output, buffers)
    truth = LF.true_solutions(buffers)
    n = length(items)
    numerator += _host_scalar(sum(abs2, predicted .- truth))
    denominator += _host_scalar(sum(abs2, truth))
    pde_total += _host_scalar(pde) * n
    bc_total += _host_scalar(bc) * n
    sup_total += _host_scalar(sup) * n
    seen += n
  end
  use_gpu && CUDA.synchronize()
  return (
    count=seen,
    relerr=sqrt(numerator / max(denominator, eps(Float64))),
    pde=pde_total / seen,
    bc=bc_total / seen,
    supervised=sup_total / seen,
  )
end

function evaluate_dataset_checkpoint(path::AbstractString, representation::Symbol,
                                     dataset::Dict, config;
                                     use_gpu::Bool=CUDA.functional())
  isempty(dataset) && error("Cannot evaluate an empty dataset")
  net, params, state, metadata = PINN.SafeTensorSnapshots.load_any_model(path)
  Symbol(metadata["representation"]) === representation ||
    error("Representation mismatch in $path")
  settings = _settings(representation, Dict(first(dataset)), config)
  params_device = use_gpu ? CUDA.cu(params) : params
  state_device = use_gpu ? CUDA.cu(state) : state
  to_device = use_gpu ? CUDA.cu : identity
  items = collect(dataset)
  buffers = representation === :eigenvalue ?
    LF.precompute_batch_eig_buffers(settings, items, use_gpu, to_device) :
    LF.precompute_batch_buffers(settings, items, use_gpu, to_device)
  output = first(net(buffers.X, params_device, state_device))
  pde, bc, sup = representation === :eigenvalue ?
    LF.batched_eigenvalue_losses(output, buffers) :
    LF.batched_power_series_losses(output, buffers; variation=representation === :power_series_variation)
  predicted = LF.batched_reconstruct(output, buffers)
  truth = LF.true_solutions(buffers)
  predicted_cpu = Array(predicted)
  truth_cpu = Array(truth)
  errors = [LF.relative_l2(predicted_cpu[:, i], truth_cpu[:, i])
            for i in axes(predicted_cpu, 2)]
  points = [LF.tau_delta_from_alpha(vec(key)) for (key, _) in items]
  return (
    count=length(items),
    relerr=LF.relative_l2(predicted_cpu, truth_cpu),
    pde=_host_scalar(pde),
    bc=_host_scalar(bc),
    supervised=_host_scalar(sup),
    errors=Float64.(errors),
    points=[Dict("tau" => Float64(point[1]), "delta" => Float64(point[2])) for point in points],
    predicted=predicted_cpu,
    truth=truth_cpu,
  )
end

function _validate_candidate(metadata, config, scope, representation, region)
  training = config["training"]
  expected = Dict(
    "representation" => String(representation),
    "scope" => String(scope),
    "family" => region === nothing ? "all" : String(region),
    "batch_size" => Int(training["batch_size"]),
    "model_seed" => Int(training["model_seed"]),
    "dataset_id" => config["splits"]["train"]["dataset_id"],
  )
  for (key, value) in expected
    get(metadata, key, nothing) == value ||
      error("Checkpoint metadata mismatch for $key: $(get(metadata, key, nothing)) != $value")
  end
end

function _freeze_selected_checkpoint(root, source, scope, representation, region)
  relative = scope === :global ?
    joinpath("models", "selected", "global", "$(representation).checkpoint") :
    joinpath("models", "selected", "family", String(region), "$(representation).checkpoint")
  target = joinpath(root, relative)
  source_hash = _sha256_file(source)
  if isfile(target)
    _sha256_file(target) == source_hash || error(
      "Frozen selected model already exists with different contents: $target")
  else
    mkpath(dirname(target))
    cp(source, target)
  end
  return relative, source_hash
end

function _verify_frozen_selection(root, selection)
  get(selection, "selection_complete", false) == true ||
    error("Checkpoint selection is incomplete")
  models = get(selection, "models", Dict())
  length(models) == 14 || error("Expected 14 selected models, found $(length(models))")
  for (key, selected) in models
    checkpoint = joinpath(root, selected["checkpoint"])
    isfile(checkpoint) || error("Missing frozen selected checkpoint for $key: $checkpoint")
    _sha256_file(checkpoint) == selected["checkpoint_sha256"] ||
      error("Frozen selected checkpoint changed for $key: $checkpoint")
  end
  return true
end

function select_checkpoints(root::AbstractString; evaluation_batch_size::Int=16_384,
                            verify_datasets::Bool=true, use_gpu::Bool=CUDA.functional())
  existing_path = joinpath(root, "selection", "selected-checkpoints.json")
  if isfile(existing_path)
    existing = JSON.parsefile(existing_path)
    _verify_frozen_selection(root, existing)
    @info "Reusing frozen checkpoint selection" path=existing_path
    return existing
  end
  study = load_study(root; verify_datasets=verify_datasets)
  config = study.config
  validation = study.splits[:validation]
  training = config["training"]
  checkpoint_every = Int(training["checkpoint_every_epochs"])
  epochs = Int(training["epochs"])
  rows = NamedTuple[]
  selected = Dict{String,Any}()

  jobs = [(scope=:global, region=nothing, representation=rep) for rep in representations(study.config)]
  append!(jobs, [(scope=:family, region=region, representation=rep)
                 for region in REGIONS for rep in representations(study.config)])
  for job in jobs
    dir = _model_dir(root, job.scope, job.representation, job.region)
    candidates = _candidate_checkpoints(dir, checkpoint_every, epochs)
    candidate_rows = NamedTuple[]
    for path in candidates
      metadata = _candidate_metadata(path)
      _validate_candidate(metadata, config, job.scope, job.representation, job.region)
      metrics = evaluate_checkpoint(path, job.representation, validation, config;
        region=job.region, batch_size=evaluation_batch_size, use_gpu=use_gpu)
      valid = _metrics_are_finite(metrics)
      row = (
        model=_model_key(job.scope, job.representation, job.region),
        scope=String(job.scope),
        family=job.region === nothing ? "all" : String(job.region),
        representation=String(job.representation),
        epoch=Int(metadata["epoch"]),
        optimizer_updates=Int(metadata["optimizer_updates"]),
        checkpoint=relpath(path, root),
        valid=valid,
        count=metrics.count,
        relerr=metrics.relerr,
        pde=metrics.pde,
        bc=metrics.bc,
        supervised=metrics.supervised,
      )
      push!(candidate_rows, row)
      push!(rows, row)
      @info "Validation checkpoint evaluated" model=row.model epoch=row.epoch relerr=row.relerr
    end
    valid_rows = filter(row -> row.valid, candidate_rows)
    isempty(valid_rows) && error(
      "No finite validation checkpoint exists for $(_model_key(job.scope, job.representation, job.region))")
    winner = sort(valid_rows; by=row -> (row.relerr, row.epoch))[1]
    winner_path = joinpath(root, winner.checkpoint)
    frozen_path, frozen_hash = _freeze_selected_checkpoint(
      root, winner_path, job.scope, job.representation, job.region)
    selected[winner.model] = Dict(
      "checkpoint" => frozen_path,
      "checkpoint_sha256" => frozen_hash,
      "training_checkpoint" => winner.checkpoint,
      "epoch" => winner.epoch,
      "optimizer_updates" => winner.optimizer_updates,
      "selection_metric" => "validation_solution_relative_l2",
      "validation" => Dict(
        "dataset_id" => dataset_id(validation),
        "count" => winner.count,
        "relerr" => winner.relerr,
        "pde" => winner.pde,
        "bc" => winner.bc,
        "supervised" => winner.supervised,
      ),
    )
  end

  selection_dir = joinpath(root, "selection")
  mkpath(selection_dir)
  CSV.write(joinpath(selection_dir, "validation-metrics.csv"), DataFrame(rows))
  payload = Dict(
    "format" => "polymathjr-checkpoint-selection",
    "format_version" => 1,
    "selection_complete" => true,
    "test_split_used" => false,
    "validation_dataset_id" => dataset_id(validation),
    "criterion" => "minimum solution-space relative L2; earliest epoch breaks exact ties",
    "models" => selected,
  )
  open(joinpath(selection_dir, "selected-checkpoints.json"), "w") do io
    JSON.print(io, payload, 2)
  end
  return payload
end

function _transfer_panel(id, title, matrix)
  Dict(
    "id" => id,
    "kind" => "heatmap",
    "title" => title,
    "xlabel" => "tested on",
    "ylabel" => "trained on",
    "opts" => Dict("log_color" => true, "floor_val" => 1e-20,
                   "yflip" => true, "annotate" => true),
    "data" => Dict(
      "matrix" => [collect(row) for row in eachrow(matrix)],
      "row_labels" => String.(REGIONS),
      "col_labels" => String.(REGIONS),
      "x_values" => nothing,
      "y_values" => nothing,
    ),
  )
end

function _write_transfer(root, representation, matrices, test)
  panels = [
    _transfer_panel("transfer_relerr", "Solution rel-L2 transfer ($(representation))", matrices.relerr),
    _transfer_panel("transfer_pde", "PDE residual component", matrices.pde),
    _transfer_panel("transfer_bc", "IC component", matrices.bc),
    _transfer_panel("transfer_sup", "Supervised component", matrices.supervised),
  ]
  payload = Dict(
    "name" => "10M selected-checkpoint transfer ($(representation))",
    "meta" => Dict(
      "shape" => "transfer",
      "representation" => String(representation),
      "dataset_id" => dataset_id(test),
      "n_per_region" => [family_size(test, region) for region in REGIONS],
      "checkpoint_policy" => "validation-selected",
      "train_regions" => String.(REGIONS),
      "test_regions" => String.(REGIONS),
    ),
    "panels" => panels,
  )
  data_dir = joinpath(root, "data")
  mkpath(data_dir)
  path = joinpath(data_dir, "batch_transfer_$(representation).json")
  open(path, "w") do io
    JSON.print(io, payload, 2)
  end
end

function run_final_test(root::AbstractString; evaluation_batch_size::Int=16_384,
                        verify_datasets::Bool=true, use_gpu::Bool=CUDA.functional())
  selection_path = joinpath(root, "selection", "selected-checkpoints.json")
  isfile(selection_path) || error("Run validation selection before final test: $selection_path")
  selection = JSON.parsefile(selection_path)
  _verify_frozen_selection(root, selection)
  existing_path = joinpath(root, "data", "final-test-metrics.json")
  if isfile(existing_path)
    existing = JSON.parsefile(existing_path)
    get(existing, "format", nothing) == "polymathjr-final-test" ||
      error("Unsupported existing final-test results: $existing_path")
    @info "Reusing frozen final-test results" path=existing_path
    return existing
  end
  study = load_study(root; verify_datasets=verify_datasets)
  test = study.splits[:test]
  config = study.config
  global_metrics = Dict{String,Any}()

  for representation in representations(study.config)
    key = _model_key(:global, representation, nothing)
    selected = selection["models"][key]
    checkpoint = joinpath(root, selected["checkpoint"])
    _sha256_file(checkpoint) == selected["checkpoint_sha256"] ||
      error("Selected checkpoint changed after validation: $checkpoint")
    by_region = Dict{String,Any}()
    all_metrics = evaluate_checkpoint(checkpoint, representation, test, config;
      batch_size=evaluation_batch_size, use_gpu=use_gpu)
    _assert_finite_metrics(all_metrics, "global/$representation on all final-test regions")
    by_region["all"] = Dict(String(k) => v for (k, v) in pairs(all_metrics))
    for region in REGIONS
      region_metrics = evaluate_checkpoint(checkpoint, representation, test, config;
        region=region, batch_size=evaluation_batch_size, use_gpu=use_gpu)
      _assert_finite_metrics(region_metrics, "global/$representation on $region final test")
      by_region[String(region)] = Dict(String(k) => v for (k, v) in pairs(region_metrics))
    end
    global_metrics[String(representation)] = by_region

    rel = zeros(Float64, 6, 6)
    pde = similar(rel); bc = similar(rel); sup = similar(rel)
    for (i, train_region) in enumerate(REGIONS)
      family_key = _model_key(:family, representation, train_region)
      family_selected = selection["models"][family_key]
      family_checkpoint = joinpath(root, family_selected["checkpoint"])
      _sha256_file(family_checkpoint) == family_selected["checkpoint_sha256"] ||
        error("Selected checkpoint changed after validation: $family_checkpoint")
      for (j, test_region) in enumerate(REGIONS)
        metrics = evaluate_checkpoint(family_checkpoint, representation, test, config;
          region=test_region, batch_size=evaluation_batch_size, use_gpu=use_gpu)
        _assert_finite_metrics(metrics,
          "family/$train_region/$representation on $test_region final test")
        rel[i, j] = metrics.relerr
        pde[i, j] = metrics.pde
        bc[i, j] = metrics.bc
        sup[i, j] = metrics.supervised
      end
    end
    _write_transfer(root, representation,
      (relerr=rel, pde=pde, bc=bc, supervised=sup), test)
  end

  payload = Dict(
    "format" => "polymathjr-final-test",
    "format_version" => 1,
    "dataset_id" => dataset_id(test),
    "count" => test.size,
    "checkpoint_policy" => "validation-selected and checksum-frozen",
    "global_models" => global_metrics,
  )
  data_dir = joinpath(root, "data")
  mkpath(data_dir)
  open(joinpath(data_dir, "final-test-metrics.json"), "w") do io
    JSON.print(io, payload, 2)
  end
  return payload
end

function _points_from_dataset(dataset::Dict)
  points = [LF.tau_delta_from_alpha(vec(key)) for (key, _) in dataset]
  sort!(points; by=point -> (point[1], point[2]))
  return [Dict("tau" => Float64(point[1]), "delta" => Float64(point[2]))
          for point in points]
end


function _prepare_inference_points(root::AbstractString, config;
                                   Rmax::Int=4, shell_count::Int=1_000,
                                   grid_size::Int=81, map_limit::Float32=4f0,
                                   seed::Int=1234)
  inference_dir = joinpath(root, "datasets", "inference")
  mkpath(inference_dir)
  path = joinpath(inference_dir, "points.json")
  if isfile(path)
    payload = JSON.parsefile(path)
    get(payload, "format", nothing) == "polymathjr-inference-points" ||
      error("Unsupported inference point file: $path")
    return payload
  end
  N = Int(config["training"]["power_series_degree"])
  shells = Dict{String,Any}()
  for (region_index, region) in enumerate(REGIONS), radius in 1:Rmax
    dataset = Plugboard.generate_shell_dataset(
      region, radius, shell_count, N;
      rng=MersenneTwister(seed + 10_000 * region_index + radius))
    shells["$(region)/$(radius)"] = _points_from_dataset(dataset)
  end
  axis = Float32.(range(-map_limit, map_limit, length=grid_size))
  grid = [Dict("tau" => Float64(tau), "delta" => Float64(delta))
          for delta in axis for tau in axis]
  showcase = [
    Dict("name" => "memorize", "tau" => 0.5, "delta" => -0.5),
    Dict("name" => "in-family", "tau" => 1.2, "delta" => 0.3),
    Dict("name" => "out-range-a", "tau" => 2.5, "delta" => -1.0),
    Dict("name" => "out-range-b", "tau" => -2.5, "delta" => 1.5),
    Dict("name" => "out-family-c", "tau" => 0.5, "delta" => 1.0),
    Dict("name" => "out-family-d", "tau" => 0.0, "delta" => 1.0),
  ]
  payload = Dict(
    "format" => "polymathjr-inference-points",
    "format_version" => 1,
    "seed" => seed,
    "power_series_degree" => N,
    "Rmax" => Rmax,
    "shell_count" => shell_count,
    "grid_size" => grid_size,
    "map_limit" => map_limit,
    "shells" => shells,
    "grid" => grid,
    "showcase" => showcase,
  )
  open(path, "w") do io
    JSON.print(io, payload)
  end
  return payload
end

function _dataset_from_points(points, N::Int)
  taus = Float32[point["tau"] for point in points]
  deltas = Float32[point["delta"] for point in points]
  return Plugboard.generate_grid_dataset(taus, deltas, N)
end

function _selected_checkpoint(root, selection, key)
  selected = selection["models"][key]
  path = joinpath(root, selected["checkpoint"])
  _sha256_file(path) == selected["checkpoint_sha256"] ||
    error("Selected checkpoint changed after validation: $path")
  return path
end

function _metric_dict(metrics; include_errors::Bool=false, include_curves::Bool=false)
  _assert_finite_metrics(metrics, "frozen generalization suite")
  if include_errors
    all(isfinite, metrics.errors) ||
      error("Non-finite per-point errors in frozen generalization suite")
  end
  if include_curves
    all(isfinite, metrics.predicted) || error("Non-finite predicted showcase curve")
    all(isfinite, metrics.truth) || error("Non-finite true showcase curve")
  end
  result = Dict(
    "count" => metrics.count,
    "relerr" => metrics.relerr,
    "pde" => metrics.pde,
    "bc" => metrics.bc,
    "supervised" => metrics.supervised,
  )
  if include_errors
    result["errors"] = metrics.errors
    result["points"] = metrics.points
    thresholds = 10.0 .^ (-4:0)
    result["coverage_fraction"] = Dict(
      string(threshold) => count(error -> error <= threshold, metrics.errors) / length(metrics.errors)
      for threshold in thresholds)
  end
  if include_curves
    size(metrics.predicted, 2) == 1 || error("Curve export requires exactly one ODE")
    result["predicted"] = Float64.(vec(metrics.predicted))
    result["truth"] = Float64.(vec(metrics.truth))
  end
  return result
end

"""
Run every frozen-model generalization experiment without retraining:
extrapolation, cross-region range, showcase, and global/family error maps.
Transfer is produced by `run_final_test` from the untouched materialized test
split. Exact shell, grid, and showcase inputs are persisted under
`datasets/inference/points.json` and reused on subsequent inference runs.
"""
function run_generalization_suite(root::AbstractString;
                                  Rmax::Int=4, shell_count::Int=1_000,
                                  grid_size::Int=81, map_limit::Float32=4f0,
                                  seed::Int=1234, use_gpu::Bool=CUDA.functional())
  selection_path = joinpath(root, "selection", "selected-checkpoints.json")
  isfile(selection_path) || error("Run validation selection first: $selection_path")
  selection = JSON.parsefile(selection_path)
  _verify_frozen_selection(root, selection)
  existing_path = joinpath(root, "data", "generalization-suite.json")
  if isfile(existing_path)
    existing = JSON.parsefile(existing_path)
    get(existing, "format", nothing) == "polymathjr-frozen-generalization-suite" &&
      get(existing, "format_version", nothing) == 2 ||
      error("Unsupported existing generalization suite: $existing_path")
    @info "Reusing frozen generalization suite" path=existing_path
    return existing
  end
  study = load_study(root; verify_datasets=false)
  points = _prepare_inference_points(root, study.config;
    Rmax=Rmax, shell_count=shell_count, grid_size=grid_size,
    map_limit=map_limit, seed=seed)
  Int(points["Rmax"]) == Rmax || error("Existing inference points use a different Rmax")
  Int(points["grid_size"]) == grid_size || error("Existing inference points use a different grid size")
  N = Int(study.config["training"]["power_series_degree"])
  results = Dict{String,Any}()

  for representation in representations(study.config)
    rep_results = Dict{String,Any}()
    family_checkpoints = Dict(region => _selected_checkpoint(root, selection,
      _model_key(:family, representation, region)) for region in REGIONS)
    global_checkpoint = _selected_checkpoint(root, selection,
      _model_key(:global, representation, nothing))

    extrapolation = Dict{String,Any}()
    range_results = Dict{String,Any}()
    for train_region in REGIONS
      own = Dict{String,Any}()
      cross = Dict{String,Any}()
      for test_region in REGIONS, radius in 1:Rmax
        key = "$(test_region)/$(radius)"
        dataset = _dataset_from_points(points["shells"][key], N)
        metrics = evaluate_dataset_checkpoint(family_checkpoints[train_region],
          representation, dataset, study.config; use_gpu=use_gpu)
        cross[key] = _metric_dict(metrics)
        train_region == test_region && (own[string(radius)] = _metric_dict(metrics))
      end
      extrapolation[String(train_region)] = own
      range_results[String(train_region)] = cross
    end
    rep_results["extrapolation"] = extrapolation
    rep_results["cross_region_range"] = range_results

    showcase_results = Dict{String,Any}()
    for scenario in points["showcase"]
      dataset = _dataset_from_points([scenario], N)
      metrics = evaluate_dataset_checkpoint(global_checkpoint, representation,
        dataset, study.config; use_gpu=use_gpu)
      showcase_results[scenario["name"]] = _metric_dict(metrics; include_curves=true)
    end
    rep_results["showcase"] = showcase_results

    grid_dataset = _dataset_from_points(points["grid"], N)
    global_grid = evaluate_dataset_checkpoint(global_checkpoint, representation,
      grid_dataset, study.config; use_gpu=use_gpu)
    maps = Dict("global" => _metric_dict(global_grid; include_errors=true))
    for region in REGIONS
      region_grid = evaluate_dataset_checkpoint(family_checkpoints[region],
        representation, grid_dataset, study.config; use_gpu=use_gpu)
      maps[String(region)] = _metric_dict(region_grid; include_errors=true)
    end
    rep_results["generalization_radius"] = maps
    results[String(representation)] = rep_results
  end

  payload = Dict(
    "format" => "polymathjr-frozen-generalization-suite",
    "format_version" => 2,
    "checkpoint_policy" => "validation-selected and checksum-frozen",
    "inference_points" => relpath(joinpath(root, "datasets", "inference", "points.json"), root),
    "solution_x" => Float64.(range(0.0, 1.0,
      length=Int(study.config["training"]["collocation_points"]))),
    "experiments" => ["transfer", "extrapolation", "cross_region_range",
                      "showcase", "generalization_radius"],
    "results" => results,
  )
  data_dir = joinpath(root, "data")
  mkpath(data_dir)
  open(joinpath(data_dir, "generalization-suite.json"), "w") do io
    JSON.print(io, payload)
  end
  return payload
end

"""
Evaluate arbitrary trace-determinant points with every frozen selected model.

`points` is a JSON-compatible vector of objects containing finite `tau` and
`delta` values. The output carries per-point solution curves, truth curves,
relative errors, and aggregate loss components for all 14 selected models.
"""
function run_custom_inference(root::AbstractString, points::AbstractVector,
                              output_path::AbstractString;
                              use_gpu::Bool=CUDA.functional())
  isempty(points) && error("Custom inference requires at least one point")
  normalized = [Dict(
    "tau" => Float64(point["tau"]),
    "delta" => Float64(point["delta"]),
    "name" => String(get(point, "name", "point-$index")),
  ) for (index, point) in enumerate(points)]
  all(point -> isfinite(point["tau"]) && isfinite(point["delta"]), normalized) ||
    error("Custom inference points must be finite")
  coordinates = [(point["tau"], point["delta"]) for point in normalized]
  length(unique(coordinates)) == length(coordinates) ||
    error("Custom inference points must have unique (tau, delta) coordinates")

  study = load_study(root; verify_datasets=false)
  selection_path = joinpath(root, "selection", "selected-checkpoints.json")
  isfile(selection_path) || error("Missing frozen checkpoint selection: $selection_path")
  selection = JSON.parsefile(selection_path)
  _verify_frozen_selection(root, selection)
  N = Int(study.config["training"]["power_series_degree"])
  dataset = _dataset_from_points(normalized, N)
  models = Dict{String,Any}()
  jobs = [(scope=:global, region=nothing, representation=rep) for rep in representations(study.config)]
  append!(jobs, [(scope=:family, region=region, representation=rep)
                 for region in REGIONS for rep in representations(study.config)])

  for job in jobs
    key = _model_key(job.scope, job.representation, job.region)
    checkpoint = _selected_checkpoint(root, selection, key)
    metrics = evaluate_dataset_checkpoint(checkpoint, job.representation,
      dataset, study.config; use_gpu=use_gpu)
    _assert_finite_metrics(metrics, "custom inference with $key")
    all(isfinite, metrics.errors) || error("Non-finite per-point custom inference errors for $key")
    models[key] = Dict(
      "checkpoint" => selection["models"][key]["checkpoint"],
      "checkpoint_sha256" => selection["models"][key]["checkpoint_sha256"],
      "count" => metrics.count,
      "relerr" => metrics.relerr,
      "pde" => metrics.pde,
      "bc" => metrics.bc,
      "supervised" => metrics.supervised,
      "points" => metrics.points,
      "errors" => metrics.errors,
      "predicted" => [Float64.(metrics.predicted[:, index])
                       for index in axes(metrics.predicted, 2)],
      "truth" => [Float64.(metrics.truth[:, index])
                   for index in axes(metrics.truth, 2)],
    )
  end

  payload = Dict(
    "format" => "polymathjr-custom-frozen-inference",
    "format_version" => 1,
    "study_root" => abspath(root),
    "checkpoint_policy" => "validation-selected and checksum-frozen",
    "solution_x" => Float64.(range(0.0, 1.0,
      length=Int(study.config["training"]["collocation_points"]))),
    "requested_points" => normalized,
    "models" => models,
  )
  mkpath(dirname(output_path))
  open(output_path, "w") do io
    JSON.print(io, payload)
  end
  return payload
end

end
