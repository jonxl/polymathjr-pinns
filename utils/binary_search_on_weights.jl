module ExperimentsForWeights

using Dates
using JSON

include("../utils/helper_funcs.jl")
using .helper_funcs

include("../architectures/PINN.jl")
using .PINN

"""
Perform binary search on a single weight while keeping others fixed.
Creates directories and saves results for each iteration.

Parameters:
- weight_name: Symbol indicating which weight to search (:supervised, :bc, or :pde)
- weight_range: Tuple (min, max) defining the search range
- num_iterations: Number of binary search iterations to perform
- fixed_weights: NamedTuple with the fixed values for the other weights
"""

function search(training_dataset, weight_name::Symbol, weight_range::Tuple{Int,Int},
  num_iterations::Int;
  fixed_weights::NamedTuple,
  num_supervised=5, N=5, x_left=0.0f0, x_right=1.0f0,
  xs,
  base_data_dir="data")

  left, right = weight_range

  println("Starting binary search on $(weight_name) weight")
  println("Range: [$(left), $(right)]")
  println("Fixed weights: $(fixed_weights)")
  println("="^50)

  for iteration in 1:num_iterations
    # Calculate midpoint
    mid = (left + right) / 2

    println("\nIteration $(iteration)/$(num_iterations)")
    println("Testing weight value: $(mid)")

    # Set weights based on which one we are searching
    if weight_name == :supervised
      supervised_weight = mid
      pde_weight = fixed_weights.pde
    elseif weight_name == :pde
      supervised_weight = fixed_weights.supervised
      pde_weight = mid
    else
      error("Unknown weight_name: $(weight_name)")
    end

    # Create directory for this iteration
    iteration_dir = joinpath(base_data_dir, "$(weight_name)-weight-$(round(mid, digits=4))")
    mkpath(iteration_dir)

    # Create info file with training configuration
    info_file = joinpath(iteration_dir, "training_info.txt")
    open(info_file, "w") do f
      write(f, "Binary Search Training Run\n")
      write(f, "="^50 * "\n\n")
      write(f, "Iteration: $(iteration) / $(num_iterations)\n")
      write(f, "Weight being searched: $(weight_name)\n")
      write(f, "Weight value tested: $(mid)\n\n")
      write(f, "Weight Configuration:\n")
      write(f, "  supervised_weight: $(supervised_weight)\n")
      write(f, "  pde_weight: $(pde_weight)\n\n")
      write(f, "Fixed Weights:\n")
      for (key, val) in pairs(fixed_weights)
        write(f, "  $(key): $(val)\n")
      end
      write(f, "\nOther Settings:\n")
      write(f, "  num_supervised: $(num_supervised)\n")
      write(f, "  N: $(N)\n")
      write(f, "  x_left: $(x_left)\n")
      write(f, "  x_right: $(x_right)\n")
      write(f, "  Search range: [$(left), $(right)]\n")
      write(f, "\nTimestamp: $(now())\n")
    end

    # Create data directory paths for evaluate_solution
    data_directories = [
      joinpath(iteration_dir, "function_comparison.png"),
      joinpath(iteration_dir, "coefficient_comparison.png"),
    ]

    # Train with current weight configuration
    for (run_idx, inner_dict) in training_dataset
      ConvertSettings = StringToMatrixSettings(inner_dict)
      converted_dict = convert_plugboard_keys(ConvertSettings)
      settings = PINNSettings(5, 1234, converted_dict, 500, 500,
        num_supervised, N, 10, x_left, x_right,
        supervised_weight, pde_weight, xs)
      # Train the network
      println("  Training network...")
      p_trained, coeff_net, st = train_pinn(settings)

      # Evaluate and save results to the iteration directory
      println("  Evaluating and saving results...")
      evaluate_solution(settings, p_trained, coeff_net, st,
        training_dataset["01"], data_directories)
    end

    println("  Results saved to: $(iteration_dir)")

    # Update search bounds
    # You can modify this logic based on your specific criteria
    if iteration % 2 == 0
      left = mid
    else
      right = mid
    end

    println("Next search range: [$(left), $(right)]")
  end

  # Create summary file
  summary_file = joinpath(base_data_dir, "$(weight_name)_search_summary.txt")
  open(summary_file, "w") do f
    write(f, "Binary Search Summary: $(weight_name) weight\n")
    write(f, "="^50 * "\n\n")
    write(f, "Total iterations: $(num_iterations)\n")
    write(f, "Search range: $(weight_range)\n")
    write(f, "Fixed weights: $(fixed_weights)\n\n")
    write(f, "Results by weight value:\n")
    write(f, "-"^50 * "\n")
  end

  println("\n" * "="^50)
  println("Binary search complete!")
  println("Summary saved to: $(summary_file)")

end

export binary_search_weights

end
