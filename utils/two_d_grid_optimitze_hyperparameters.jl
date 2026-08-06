using Plots
using Statistics

"""
  GridSearchResult
  
Stores the results of a hyperparameter search.
"""

struct GridSearchResult
  weight_values::Dict{Symbol, Vector{Float64}}  # e.g., :supervised => [1.0, 2.0, ...]
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
function evaluate_weight_configuration(training_dataset, weights::NamedTuple,
  num_supervised, N, x_left, x_right, xs;
  base_data_dir="data")

  # Extract weights
  supervised_weight = weights.supervised
  pde_weight = weights.pde

  # Create directory for this configuration
  config_dir = joinpath(base_data_dir, 
                       "s$(round(supervised_weight, digits=2))-" *
                        "p$(round(pde_weight, digits=2))")
  mkpath(config_dir)

  # Initialize objective value accumulator
  total_error = 0.0
  num_runs = 0

  # Train with current weight configuration
  for (run_idx, inner_dict) in training_dataset
    ConvertSettings = StringToMatrixSettings(inner_dict)
    converted_dict = ConvertStringToMatrix.convert(ConvertSettings)
    settings = PINNSettings(5, 1234, converted_dict, 500, 500,
                           num_supervised, N, 10, x_left, x_right,
                           supervised_weight, pde_weight, xs)
    
    # Train the network
    println("  Training with weights: supervised=$(supervised_weight), pde=$(pde_weight)")
    p_trained, coeff_net, st = train_pinn(settings)
    
    # Evaluate and compute error metric
    # You will need to define what metric you want to optimize
    # For example: MSE, validation loss, etc.
    error = compute_objective_metric(settings, p_trained, coeff_net, st, inner_dict)
    
    total_error += error
    num_runs += 1
    
    # Save results
    data_directories = [
      joinpath(config_dir, "function_comparison.png"),
      joinpath(config_dir, "coefficient_comparison.png"),
    ]
    evaluate_solution(settings, p_trained, coeff_net, st, 
                     training_dataset["01"], data_directories)
  end
  
  # Return average error as objective value
  return total_error / num_runs
end

"""
  grid_search_2d(training_dataset, weight1::Symbol, weight1_range::Tuple,
                 weight2::Symbol, weight2_range::Tuple, num_points::Int;
                 fixed_weights::NamedTuple, kwargs...)

Perform 2D grid search over two hyperparameters.
"""
function grid_search_2d(training_dataset,
  weight1::Symbol, weight1_range::Tuple{Float64, Float64},
  weight2::Symbol, weight2_range::Tuple{Float64, Float64},
  num_points::Int;
  fixed_weights::NamedTuple,
  num_supervised=5, N=5, x_left=0.0f0, x_right=1.0f0,
  xs=nothing,
  base_data_dir="data/grid_search")

  println("Starting 2D grid search")
  println("Weight 1: $(weight1), Range: $(weight1_range)")
  println("Weight 2: $(weight2), Range: $(weight2_range)")
  println("Grid points per dimension: $(num_points)")
  println("Fixed weights: $(fixed_weights)")
  println("="^50)

  # Create grid of weight values
  weight1_values = range(weight1_range[1], weight1_range[2], length=num_points)
  weight2_values = range(weight2_range[1], weight2_range[2], length=num_points)

  # Initialize results matrix
  objective_matrix = zeros(Float64, num_points, num_points)

  # Track best configuration
  best_objective = Inf
  best_weights = nothing

  # Iterate through grid
  for (i, w1) in enumerate(weight1_values)
    for (j, w2) in enumerate(weight2_values)
      println("\nGrid point ($(i), $(j)) / ($(num_points), $(num_points))")

      # Create weight configuration
      weights = create_weight_tuple(weight1, w1, weight2, w2, fixed_weights)

      # Evaluate this configuration
      objective_value = evaluate_weight_configuration(
        training_dataset, weights, num_supervised, N, 
        x_left, x_right, xs; base_data_dir=base_data_dir
      )

      objective_matrix[j, i] = objective_value

      println("  Objective value: $(objective_value)")

      # Update best configuration
      if objective_value < best_objective
        best_objective = objective_value
        best_weights = weights
        println("  *** New best configuration found! ***")
      end
    end
  end
  
  # Store results
  weight_values = Dict(weight1 => collect(weight1_values), 
                      weight2 => collect(weight2_values))
  
  result = GridSearchResult(weight_values, objective_matrix, 
                           best_weights, best_objective)
  
  # Create visualizations
  visualize_search_results(result, weight1, weight2, base_data_dir)
  
  # Save summary
  save_search_summary(result, weight1, weight2, base_data_dir)
  
  return result
end

"""
  random_search_2d(training_dataset, weight1::Symbol, weight1_range::Tuple,
                   weight2::Symbol, weight2_range::Tuple, num_samples::Int;
                   fixed_weights::NamedTuple, kwargs...)

Perform random search over two hyperparameters.
"""
function random_search_2d(training_dataset,
                          weight1::Symbol, weight1_range::Tuple{Float64, Float64},
                          weight2::Symbol, weight2_range::Tuple{Float64, Float64},
                          num_samples::Int;
                          fixed_weights::NamedTuple,
                          num_supervised=5, N=5, x_left=0.0f0, x_right=1.0f0,
                          xs=nothing,
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
  weights_dict = Dict{Symbol, Float64}()
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
  compute_objective_metric(settings, p_trained, coeff_net, st, data)

Compute the objective metric for a trained model.
You need to implement this based on your specific optimization goal.
"""
function compute_objective_metric(settings, p_trained, coeff_net, st, data)
  # TODO: Implement your specific metric
  # Examples:
  # - Validation loss
  # - Test MSE
  # - Combined metric (accuracy + regularization)
  # - Physical constraint violation
  
  # Placeholder: return a random error for demonstration
  # Replace this with your actual evaluation metric
  error("You need to implement compute_objective_metric for your specific use case")
  
  # Example structure:
  # validation_loss = compute_validation_loss(settings, p_trained, coeff_net, st)
  # pde_residual = compute_pde_residual(settings, p_trained, coeff_net, st)
  # return validation_loss + 0.1 * pde_residual
end

"""
  visualize_search_results(result::GridSearchResult, 
                          weight1::Symbol, weight2::Symbol, 
                          base_data_dir::String)

Create contour plot visualization of grid search results.
"""
function visualize_search_results(result::GridSearchResult,
                                  weight1::Symbol, weight2::Symbol,
                                  base_data_dir::String)
  
  w1_values = result.weight_values[weight1]
  w2_values = result.weight_values[weight2]
  obj_matrix = result.objective_values
  
  # Create contour plot
  p = contour(w1_values, w2_values, obj_matrix,
    xlabel=String(weight1),
    ylabel=String(weight2),
    title="Hyperparameter Search: $(weight1) vs $(weight2)",
    levels=15,
    linewidth=2,
    color=:viridis,
    fill=true)
  
  # Mark the best point
  best_w1 = result.best_weights[weight1]
  best_w2 = result.best_weights[weight2]
  scatter!([best_w1], [best_w2],
    marker=:star,
    markersize=10,
    color=:red,
    label="Best (obj=$(round(result.best_objective, digits=4)))")
  
  # Save plot
  savefig(p, joinpath(base_data_dir, "hyperparameter_contour.png"))
  # Create 3D surface plot
  p3d = surface(w1_values, w2_values, obj_matrix,
    xlabel=String(weight1),
    ylabel=String(weight2),
    zlabel="Objective Value",
    title="Objective Landscape",
    camera=(45, 30),
    color=:viridis)

  scatter3d!([best_w1], [best_w2], [result.best_objective],
    marker=:star,
    markersize=8,
    color=:red,
    label="Best")
  
  savefig(p3d, joinpath(base_data_dir, "hyperparameter_surface.png"))
  
  println("\nVisualizations saved to:")
  println("  - $(joinpath(base_data_dir, "hyperparameter_contour.png"))")
  println("  - $(joinpath(base_data_dir, "hyperparameter_surface.png"))")
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
  save_search_summary(result::GridSearchResult, weight1::Symbol, 
                     weight2::Symbol, base_data_dir::String)

Save summary of grid search results.
"""
function save_search_summary(result::GridSearchResult,
                            weight1::Symbol, weight2::Symbol,
                            base_data_dir::String)
  
  summary_file = joinpath(base_data_dir, "search_summary.txt")
  open(summary_file, "w") do f
    write(f, "Grid Search Summary\n")
    write(f, "="^50 * "\n\n")
    write(f, "Search Parameters:\n")
    write(f, "  Weight 1: $(weight1)\n")
    write(f, "  Weight 2: $(weight2)\n")
    write(f, "  Grid points: $(length(result.weight_values[weight1])) x $(length(result.weight_values[weight2]))\n\n")
    write(f, "Best Configuration Found:\n")
    write(f, "  supervised: $(result.best_weights.supervised)\n")
    write(f, "  bc: $(result.best_weights.bc)\n")
    write(f, "  pde: $(result.best_weights.pde)\n")
    write(f, "  Objective value: $(result.best_objective)\n\n")
    write(f, "Statistics:\n")
    write(f, "  Min objective: $(minimum(result.objective_values))\n")
    write(f, "  Max objective: $(maximum(result.objective_values))\n")
    write(f, "  Mean objective: $(mean(result.objective_values))\n")
    write(f, "  Std objective: $(std(result.objective_values))\n")
  end
  
  println("Summary saved to: $(summary_file)")
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

