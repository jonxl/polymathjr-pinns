module TwoDGridSearchOnWeights

using Plots
using Statistics
using CUDA
using Logging
using ProgressMeter
using JSON
using Dates
using DataFrames
using CSV

include("../modelcode/PINN.jl")
using .PINN

include("../utils/gpu_utils.jl")
using .GPUUtils

include("../utils/helper_funcs.jl")
using .helper_funcs

"""
  GridSearchResult
  
Stores the results of a hyperparameter search.
"""

struct GridSearchResult
  weight_values::Dict{Symbol,Vector{Float64}}  # e.g., :supervised => [1.0, 2.0, ...]
  objective_values::Array{Float64}  # N-dimensional array of objective values
  best_weights::NamedTuple  # Best weight configuration found
  best_objective::Float64  # Best objective value
end

"""
  evaluate_weight_configuration(training_dataset, weights::NamedTuple, 
                                 num_supervised, N, x_left, x_right, xs)

Train and evaluate a single weight configuration.
Returns an objective value (lower is better).
"""
function evaluate_weight_configuration(neuron_count, training_dataset, benchmark_dataset, weights::NamedTuple,
num_supervised, N, x_left, x_right, xs, base_data_dir; progress_callback::Union{Function,Nothing}=nothing, milestone_interval::Int=1000)
  # Extract weights
  supervised_weight = weights.supervised
  bc_weight = weights.bc
  pde_weight = weights.pde

  # Initialize objective value accumulator
  total_error = 0.0
  all_loss_history = []
  all_eval_results = Dict[]
  all_milestones = NamedTuple[]

  # Train with current weight configuration
  for (run_idx, inner_dict) in training_dataset
    converted_dict = convert_plugboard_keys(inner_dict)

    float_converted_dict = Dict{Matrix{Float32}, Any}()
    for (mat, series) in converted_dict
      float_converted_dict[Float32.(mat)] = series
    end

    settings = PINNSettings(neuron_count, 1234, Dict{Any,Any}(float_converted_dict), 10000, N, num_supervised, 1000, x_left, x_right, Float32(supervised_weight), Float32(bc_weight), Float32(pde_weight), xs, "adam")

    # Milestone callback: evaluate at intermediate checkpoints
    milestones = NamedTuple[]
    on_milestone = function(p_current, iteration, coeff_net, st, run_id)
      loss, eval_results = evaluate_solution(settings, p_current, coeff_net, st,
        benchmark_dataset["01"], base_data_dir, run_id;
        write_results_json=false)
      coeffs = length(eval_results) > 0 ? eval_results[end]["pinn_coefficients"] : Float64[]
      push!(milestones, (iteration=iteration, objective=Float64(loss), coefficients=coeffs))
    end

    # Train the network — no per-config I/O
    p_trained, coeff_net, st, run_id, history = train_pinn(settings, base_data_dir;
      write_loss_csv=false, progress_callback=progress_callback,
      milestone_interval=milestone_interval, on_milestone=on_milestone)

    # Final evaluation — no per-config I/O
    function_error, eval_results = evaluate_solution(settings, p_trained, coeff_net, st,
      benchmark_dataset["01"], base_data_dir, run_id;
      write_results_json=false)

    total_error += function_error
    push!(all_loss_history, history)
    append!(all_eval_results, eval_results)
    append!(all_milestones, milestones)
  end

  return (total_error, all_loss_history, all_eval_results, all_milestones)
end

"""
  estimate_batch_size()

Estimate how many concurrent PINNs can run in parallel based on VRAM
capacity and available threads. Falls back to thread count on CPU.
"""
function estimate_batch_size()
  if !GPUUtils.is_gpu_available()
    return Threads.nthreads()
  end

  available_vram = CUDA.available_memory()
  per_pinn_estimate = 50 * 1024 * 1024  # ~50MB conservative (params + optimizer + AD tape)
  gpu_batch = max(1, Int(floor(available_vram / per_pinn_estimate)))

  # Don't exceed thread count — no point spawning more tasks than threads
  return min(gpu_batch, Threads.nthreads())
end

"""
  grid_search_2d(training_dataset, weight1::Symbol, weight1_range::Tuple,
                 weight2::Symbol, weight2_range::Tuple, num_points::Int;
                 fixed_weights::NamedTuple, kwargs...)

Perform 2D grid search over two hyperparameters.
"""
function grid_search_2d(neuron_count, training_dataset, benchmark_dataset,
  weight1::Symbol, weight1_range::Tuple{Float64,Float64},
  weight2::Symbol, weight2_range::Tuple{Float64,Float64},
  num_points::Int;
  fixed_weights::NamedTuple,
  num_supervised, N, x_left, x_right,
  xs,
  base_data_dir,
  milestone_interval::Int=1000)

  # ---- Startup Banner ----
  gpu_name = GPUUtils.is_gpu_available() ? GPUUtils.get_device() : "none (CPU only)"
  vram_str = GPUUtils.is_gpu_available() ? "$(round(CUDA.available_memory() / 1024^3, digits=2)) GB free" : "N/A"

  println("="^60)
  println("  2D GRID SEARCH — THREADED")
  println("="^60)
  println("  Threads:    $(Threads.nthreads())")
  println("  GPU:        $(gpu_name)")
  println("  VRAM:       $(vram_str)")
  println("  Grid:       $(num_points) x $(num_points) = $(num_points^2) configs")
  println("  Weight 1:   $(weight1) in $(weight1_range)")
  println("  Weight 2:   $(weight2) in $(weight2_range)")
  println("  Fixed:      $(fixed_weights)")
  println("="^60)

  # Create grid of weight values
  weight1_values = range(weight1_range[1], weight1_range[2], length=num_points)
  weight2_values = range(weight2_range[1], weight2_range[2], length=num_points)

  # Initialize results matrix
  objective_matrix = zeros(Float64, num_points, num_points)

  # Flatten 2D grid into list of all configurations
  all_configs = [(i, w1, j, w2)
    for (i, w1) in enumerate(weight1_values)
    for (j, w2) in enumerate(weight2_values)]

  # Thread-safe progress counter
  total_configs = length(all_configs)
  completed = Threads.Atomic{Int}(0)

  # Pre-allocate result storage vectors
  all_loss_histories = Vector{Any}(undef, total_configs)
  all_config_results = Vector{Any}(undef, total_configs)
  all_milestones = Vector{Any}(undef, total_configs)
  all_weights = Vector{NamedTuple}(undef, total_configs)

  # Determine batch size based on VRAM capacity and thread count
  batch_size = min(total_configs, estimate_batch_size())
  num_batches = ceil(Int, total_configs / batch_size)
  println("  Batch size: $(batch_size) configs/batch ($(num_batches) batches)")
  println("="^60)

  # maxiters per PINN (must match the value in evaluate_weight_configuration)
  iters_per_pinn = 10000

  # Process grid in batched parallel chunks
  for (batch_num, batch) in enumerate(Iterators.partition(all_configs, batch_size))
    # Shared progress bar for the entire batch — all threads advance it together
    total_batch_iters = length(batch) * iters_per_pinn
    batch_bar = Progress(total_batch_iters, desc="Batch $(batch_num)/$(num_batches) [$(length(batch)) PINNs]  ")

    # Thread-safe callback: each PINN's iteration ticks the shared bar
    shared_callback = function(state, l)
      ProgressMeter.next!(batch_bar; showvalues=[(:loss, round(l, digits=6)), (:thread, Threads.threadid())])
      return false
    end

    tasks = map(batch) do (i, w1, j, w2)
      Threads.@spawn begin
        weights = create_weight_tuple(weight1, w1, weight2, w2, fixed_weights)

        total_error, loss_history, eval_results, milestones = evaluate_weight_configuration(
          neuron_count, training_dataset, benchmark_dataset, weights,
          num_supervised, N, x_left, x_right, xs, base_data_dir;
          progress_callback=shared_callback,
          milestone_interval=milestone_interval
        )

        done = Threads.atomic_add!(completed, 1) + 1
        # Compute linear index for pre-allocated storage
        linear_idx = (i - 1) * num_points + j
        (j, i, total_error, loss_history, eval_results, weights, milestones, linear_idx)
      end
    end

    # Collect results from this batch
    for task in tasks
      j, i, objective_value, loss_history, eval_results, weights, milestones, linear_idx = fetch(task)
      objective_matrix[j, i] = objective_value
      all_loss_histories[linear_idx] = loss_history
      all_config_results[linear_idx] = eval_results
      all_milestones[linear_idx] = milestones
      all_weights[linear_idx] = weights
    end

    # Log batch completion
    done_so_far = completed[]
    pct = round(100.0 * done_so_far / total_configs, digits=1)
    println("\n  Batch $(batch_num) complete — $(done_so_far)/$(total_configs) configs done ($(pct)%)")
  end

  # Post-hoc reduction: find best from completed matrix (no shared mutable state)
  best_idx = argmin(objective_matrix)
  j_best, i_best = Tuple(best_idx)
  best_w1 = weight1_values[i_best]
  best_w2 = weight2_values[j_best]
  best_weights = create_weight_tuple(weight1, best_w1, weight2, best_w2, fixed_weights)
  best_objective = objective_matrix[best_idx]

  # Store results
  weight_values = Dict(weight1 => collect(weight1_values),
    weight2 => collect(weight2_values))

  result = GridSearchResult(weight_values, objective_matrix,
    best_weights, best_objective)

  # Consolidated output — write all data at once
  write_grid_results_json(result, all_config_results, all_milestones, all_weights,
    weight1, weight2, weight1_range, weight2_range, fixed_weights,
    neuron_count, N, num_supervised, x_left, x_right, num_points,
    base_data_dir)
  write_top_loss_csvs(all_loss_histories, all_weights, objective_matrix,
    weight1_values, weight2_values, base_data_dir)

  return result
end

"""
  write_grid_results_json(result, all_config_results, all_milestones, all_weights,
                          weight1, weight2, weight1_range, weight2_range,
                          fixed_weights, neuron_count, N, num_supervised,
                          x_left, x_right, grid_size, base_data_dir)

Write consolidated grid search results to a single JSON file.
"""
function write_grid_results_json(result::GridSearchResult, all_config_results, all_milestones, all_weights,
  weight1::Symbol, weight2::Symbol,
  weight1_range::Tuple{Float64,Float64}, weight2_range::Tuple{Float64,Float64},
  fixed_weights::NamedTuple,
  neuron_count, N, num_supervised, x_left, x_right, grid_size,
  base_data_dir)

  mkpath(base_data_dir)

  w1_values = result.weight_values[weight1]
  w2_values = result.weight_values[weight2]
  obj_matrix = result.objective_values

  # Extract ode_matrix and benchmark_coefficients from first config's eval results
  ode_matrix = Float64[]
  benchmark_coefficients = Float64[]
  if length(all_config_results) > 0 && length(all_config_results[1]) > 0
    first_result = all_config_results[1][1]
    if haskey(first_result, "alpha_matrix")
      ode_matrix = Float64.(first_result["alpha_matrix"])
    end
    if haskey(first_result, "benchmark_coefficients")
      benchmark_coefficients = Float64.(first_result["benchmark_coefficients"])
    end
  end

  # Build fixed_weights dict for JSON
  fixed_weights_dict = Dict{String,Float64}()
  for (k, v) in pairs(fixed_weights)
    if k != weight1 && k != weight2
      fixed_weights_dict[String(k)] = v
    end
  end

  # Build metadata
  metadata = Dict(
    "timestamp" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
    "grid_size" => grid_size,
    "neuron_count" => neuron_count,
    "maxiters" => 10000,
    "optimizer" => "adam",
    "seed" => 1234,
    "N" => N,
    "num_supervised" => num_supervised,
    "domain" => [Float64(x_left), Float64(x_right)],
    "weight1" => Dict("name" => String(weight1), "range" => [weight1_range[1], weight1_range[2]]),
    "weight2" => Dict("name" => String(weight2), "range" => [weight2_range[1], weight2_range[2]]),
    "fixed_weights" => fixed_weights_dict,
    "ode_matrix" => ode_matrix,
    "benchmark_coefficients" => benchmark_coefficients
  )

  # Build grid section
  grid = Dict(
    "weight1_values" => collect(w1_values),
    "weight2_values" => collect(w2_values),
    "objective_matrix" => [obj_matrix[j, :] for j in 1:size(obj_matrix, 1)]
  )

  # Build configs array
  configs = []
  num_w2 = length(w2_values)
  for (idx, (i, w1)) in enumerate(zip(1:length(w1_values), w1_values))
    for (j, w2) in zip(1:num_w2, w2_values)
      linear_idx = (i - 1) * num_w2 + j
      weights = all_weights[linear_idx]
      eval_results = all_config_results[linear_idx]
      milestones_data = all_milestones[linear_idx]

      # Extract final coefficients from eval results
      coefficients = Float64[]
      if length(eval_results) > 0
        coefficients = eval_results[end]["pinn_coefficients"]
      end

      # Format milestones
      formatted_milestones = [
        Dict(
          "iteration" => m.iteration,
          "objective" => m.objective,
          "coefficients" => m.coefficients
        ) for m in milestones_data
      ]

      config_entry = Dict(
        "idx" => linear_idx - 1,
        "i" => i - 1,
        "j" => j - 1,
        "weights" => Dict(
          "supervised" => weights.supervised,
          "bc" => weights.bc,
          "pde" => weights.pde
        ),
        "objective" => obj_matrix[j, i],
        "coefficients" => coefficients,
        "milestones" => formatted_milestones
      )
      push!(configs, config_entry)
    end
  end

  output = Dict(
    "metadata" => metadata,
    "grid" => grid,
    "configs" => configs
  )

  output_file = joinpath(base_data_dir, "grid_results.json")
  open(output_file, "w") do io
    JSON.print(io, output, 2)
  end

  println("Grid results written to: $(output_file)")
end

"""
  write_top_loss_csvs(all_loss_histories, all_weights, objective_matrix,
                      weight1_values, weight2_values, base_data_dir; top_n=10)

Write loss CSVs for the top N configurations (lowest objective) to top_configs/.
"""
function write_top_loss_csvs(all_loss_histories, all_weights, objective_matrix,
  weight1_values, weight2_values, base_data_dir; top_n::Int=10)

  num_w2 = length(weight2_values)

  # Build (linear_idx, objective) pairs for all configs
  config_objectives = Tuple{Int,Float64}[]
  for (i, _) in enumerate(weight1_values)
    for (j, _) in enumerate(weight2_values)
      linear_idx = (i - 1) * num_w2 + j
      push!(config_objectives, (linear_idx, objective_matrix[j, i]))
    end
  end

  # Sort by objective (ascending = best first) and take top N
  sort!(config_objectives, by=x -> x[2])
  top_configs = config_objectives[1:min(top_n, length(config_objectives))]

  top_dir = joinpath(base_data_dir, "top_configs")
  mkpath(top_dir)

  for (rank, (linear_idx, _)) in enumerate(top_configs)
    weights = all_weights[linear_idx]
    loss_histories = all_loss_histories[linear_idx]

    # loss_histories is a vector of history arrays (one per training run)
    # Concatenate all histories for this config
    all_entries = []
    for history in loss_histories
      append!(all_entries, history)
    end

    if isempty(all_entries)
      continue
    end

    df = DataFrame(
      iteration = 1:length(all_entries),
      total = [e.total for e in all_entries],
      bc = [e.bc for e in all_entries],
      pde = [e.pde for e in all_entries],
      supervised = [e.supervised for e in all_entries]
    )

    rank_str = lpad(rank, 2, '0')
    s = round(weights.supervised, digits=2)
    b = round(weights.bc, digits=2)
    p = round(weights.pde, digits=2)
    filename = "rank$(rank_str)_s$(s)-b$(b)-p$(p).csv"
    CSV.write(joinpath(top_dir, filename), df)
  end

  println("Top $(min(top_n, length(top_configs))) loss CSVs written to: $(top_dir)")
end

"""
  random_search_2d(training_dataset, weight1::Symbol, weight1_range::Tuple,
                   weight2::Symbol, weight2_range::Tuple, num_samples::Int;
                   fixed_weights::NamedTuple, kwargs...)

Perform random search over two hyperparameters.
"""
function random_search_2d(training_dataset,
  weight1::Symbol, weight1_range::Tuple{Float64,Float64},
  weight2::Symbol, weight2_range::Tuple{Float64,Float64},
  num_samples::Int;
  fixed_weights::NamedTuple,
  num_supervised, N, x_left, x_right,
  xs,
  base_data_dir="data/random_search")
  println("Starting 2D random search")
  println("Weight 1: $(weight1), Range: $(weight1_range)")
  println("Weight 2: $(weight2), Range: $(weight2_range)")
  println("Number of samples: $(num_samples)")
  println("Fixed weights: $(fixed_weights)")
  println("="^50)

  # Store sampled points and their objectives
  weight1_samples = Float64[]
  weight2_samples = Float64[]
  objective_values = Float64[]

  best_objective = Inf
  best_weights = nothing

  for sample in 1:num_samples
    println("\nSample $(sample) / $(num_samples)")

    # Sample random weight values
    w1 = rand() * (weight1_range[2] - weight1_range[1]) + weight1_range[1]
    w2 = rand() * (weight2_range[2] - weight2_range[1]) + weight2_range[1]

    push!(weight1_samples, w1)
    push!(weight2_samples, w2)

    # Create weight configuration
    weights = create_weight_tuple(weight1, w1, weight2, w2, fixed_weights)

    # Evaluate this configuration
    objective_value = evaluate_weight_configuration(
      training_dataset, weights, num_supervised, N,
      x_left, x_right, xs; base_data_dir=base_data_dir
    )

    push!(objective_values, objective_value)

    println("  Weights: $(weight1)=$(w1), $(weight2)=$(w2)")
    println("  Objective value: $(objective_value)")

    # Update best configuration
    if objective_value < best_objective
      best_objective = objective_value
      best_weights = weights
      println("  *** New best configuration found! ***")
    end
  end

  # Create visualization with scattered points
  visualize_random_search(weight1_samples, weight2_samples, objective_values,
    weight1, weight2, weight1_range, weight2_range,
    base_data_dir)

  # Save summary
  save_random_search_summary(weight1_samples, weight2_samples, objective_values,
    best_weights, best_objective, weight1, weight2,
    base_data_dir)

  return (weight1_samples, weight2_samples, objective_values, best_weights, best_objective)
end

"""
  create_weight_tuple(weight1::Symbol, w1::Float64, 
                     weight2::Symbol, w2::Float64, 
                     fixed_weights::NamedTuple)

Helper function to create a NamedTuple with all three weights.
"""
function create_weight_tuple(weight1::Symbol, w1::Float64,
  weight2::Symbol, w2::Float64,
  fixed_weights::NamedTuple)

  weights_dict = Dict{Symbol,Float64}()

  # Set the two variables being searched
  weights_dict[weight1] = w1
  weights_dict[weight2] = w2

  # Add the fixed weight
  for (key, val) in pairs(fixed_weights)
    if key != weight1 && key != weight2
      weights_dict[key] = val
    end
  end

  # Ensure all three weights are present
  if !haskey(weights_dict, :supervised) ||
     !haskey(weights_dict, :bc) ||
     !haskey(weights_dict, :pde)
    error("Weight configuration incomplete. Need supervised, bc, and pde weights.")
  end

  return (supervised=weights_dict[:supervised],
    bc=weights_dict[:bc],
    pde=weights_dict[:pde])
end

"""
  visualize_random_search(w1_samples, w2_samples, objectives,
                         weight1::Symbol, weight2::Symbol,
                         w1_range, w2_range, base_data_dir::String)

Create visualization of random search results with scattered points.
"""
function visualize_random_search(w1_samples, w2_samples, objectives,
  weight1::Symbol, weight2::Symbol,
  w1_range, w2_range,
  base_data_dir::String)

  # Create scatter plot with color indicating objective value
  p = scatter(w1_samples, w2_samples,
    marker_z=objectives,
    xlabel=String(weight1),
    ylabel=String(weight2),
    title="Random Search: $(weight1) vs $(weight2)",
    color=:viridis,
    colorbar=true,
    markersize=6,
    label="",
    xlims=w1_range,
    ylims=w2_range)

  # Mark the best point
  best_idx = argmin(objectives)
  scatter!([w1_samples[best_idx]], [w2_samples[best_idx]],
    marker=:star,
    markersize=12,
    color=:red,
    label="Best (obj=$(round(objectives[best_idx], digits=4)))")

  savefig(p, joinpath(base_data_dir, "random_search_scatter.png"))

  # Optionally, interpolate to create a contour plot
  # This requires a package like Interpolations.jl

  println("\nVisualization saved to:")
  println("  - $(joinpath(base_data_dir, "random_search_scatter.png"))")
end

"""
  save_random_search_summary(w1_samples, w2_samples, objectives,
                            best_weights, best_objective,
                            weight1::Symbol, weight2::Symbol,
                            base_data_dir::String)

Save summary of random search results.
"""
function save_random_search_summary(w1_samples, w2_samples, objectives,
  best_weights, best_objective,
  weight1::Symbol, weight2::Symbol,
  base_data_dir::String)

  summary_file = joinpath(base_data_dir, "random_search_summary.txt")
  open(summary_file, "w") do f
    write(f, "Random Search Summary\n")
    write(f, "="^50 * "\n\n")
    write(f, "Search Parameters:\n")
    write(f, "  Weight 1: $(weight1)\n")
    write(f, "  Weight 2: $(weight2)\n")
    write(f, "  Number of samples: $(length(objectives))\n\n")
    write(f, "Best Configuration Found:\n")
    write(f, "  supervised: $(best_weights.supervised)\n")
    write(f, "  bc: $(best_weights.bc)\n")
    write(f, "  pde: $(best_weights.pde)\n")
    write(f, "  Objective value: $(best_objective)\n\n")
    write(f, "Statistics:\n")
    write(f, "  Min objective: $(minimum(objectives))\n")
    write(f, "  Max objective: $(maximum(objectives))\n")
    write(f, "  Mean objective: $(mean(objectives))\n")
    write(f, "  Std objective: $(std(objectives))\n")
  end

  println("Summary saved to: $(summary_file)")
end

export GridSearchResult, grid_search_2d, random_search_2d, estimate_batch_size

end
