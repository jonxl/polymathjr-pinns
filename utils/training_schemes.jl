module training_schemes
using DataFrames
using CSV
using Plots
using JSON
using Dates

include("../modelcode/PINN.jl")
using .PINN

include("../utils/helper_funcs.jl")
using .helper_funcs

include("../utils/two_d_grid_search_hyperparameters.jl")
using .TwoDGridSearchOnWeights

include("../utils/snapshot_utils.jl")
using .SnapshotUtils

struct TrainingSchemesSettings
  training_dataset::Dict{String,Dict{String,Any}}
  benchmark_dataset::Dict{String,Dict{String,Any}}
  N::Int
  num_supervised::Int
  num_points::Int
  x_left::Float32
  x_right::Float32
  supervised_weight::Float32
  bc_weight::Float32
  pde_weight::Float32
  xs::Vector{Float32}
end

# This will traing each NN on different neuron counts
function scaling_neurons(settings::TrainingSchemesSettings, neurons_counts::Dict{String,Int})
  for (filename, neuron_count) in neurons_counts
    @info "Starting training for $filename for $neuron_count neurons"
    for (run_idx, inner_dict) in settings.training_dataset
      converted_dict = convert_plugboard_keys(inner_dict)

      pinn_settings = PINNSettings(neuron_count, 1234, converted_dict, 500, settings.num_supervised, settings.N, settings.num_points, settings.x_left, settings.x_right, settings.supervised_weight, settings.bc_weight, settings.pde_weight, settings.xs, "adam")

      run_id = generate_run_id(pinn_settings.optimizer)
      output_dir = joinpath("results", "run-$run_id")
      mkpath(output_dir)

      # Train the network
      p_trained, coeff_net, st, _, _ = train_pinn(pinn_settings, output_dir; run_id=run_id)
      function_error, _ = evaluate_solution(pinn_settings, p_trained, coeff_net, st, settings.benchmark_dataset["01"], output_dir, run_id)
      @info "Function error: $function_error"
    end
  end
end

## Enough neurons, lets do iterations — train once, evaluate at milestones.
function scaling_adam(settings::TrainingSchemesSettings, maxiters::Int, milestone_interval::Int; snapshot_path::Union{String,Nothing}=nothing)
  for (run_idx, inner_dict) in settings.training_dataset
    converted_dict = convert_plugboard_keys(inner_dict)

    pinn_settings = PINNSettings(100, 1234, converted_dict, maxiters, settings.num_supervised, settings.N, settings.num_points, settings.x_left, settings.x_right, settings.supervised_weight, settings.bc_weight, settings.pde_weight, settings.xs, "adam")

    # Generate run_id upfront so the output directory is known before training
    run_id = generate_run_id(pinn_settings.optimizer)
    output_dir = joinpath("results", "run-$run_id")
    mkpath(output_dir)

    # Collect milestones in memory
    milestones = NamedTuple[]

    # Milestone callback — called mid-training every milestone_interval iterations
    function on_milestone(p_current, iteration, coeff_net, st, _run_id)
      # Create snapshot directory on first milestone call
      snapshot_dir = joinpath(output_dir, "snapshots")
      mkpath(snapshot_dir)

      # Save model weights as raw Float32 binary
      snapshot_path = joinpath(snapshot_dir, "iter-$(lpad(iteration, 7, '0')).bin")
      write(snapshot_path, Float32.(vec(p_current)))

      error, eval_results = evaluate_solution(pinn_settings, p_current, coeff_net, st, settings.benchmark_dataset["01"], output_dir, run_id; iteration=iteration, write_results_json=false)
      coeffs = length(eval_results) > 0 ? eval_results[end]["pinn_coefficients"] : Float64[]
      push!(milestones, (iteration=iteration, objective=Float64(error), coefficients=coeffs))
      @info "Milestone $iteration — error: $error"
    end

    # Train once, evaluating at each milestone interval
    p_trained, coeff_net, st, _, history = train_pinn(pinn_settings, output_dir; run_id=run_id, milestone_interval=milestone_interval, on_milestone=on_milestone, snapshot_path=snapshot_path)

    # Final evaluation
    final_error, final_results = evaluate_solution(pinn_settings, p_trained, coeff_net, st, settings.benchmark_dataset["01"], output_dir, run_id; write_results_json=false)
    final_coeffs = length(final_results) > 0 ? final_results[end]["pinn_coefficients"] : Float64[]

    # Extract metadata from eval results
    ode_matrix = Float64[]
    benchmark_coefficients = Float64[]
    if length(final_results) > 0
      if haskey(final_results[1], "alpha_matrix")
        ode_matrix = Float64.(final_results[1]["alpha_matrix"])
      end
      if haskey(final_results[1], "benchmark_coefficients")
        benchmark_coefficients = Float64.(final_results[1]["benchmark_coefficients"])
      end
    end

    # Write consolidated JSON
    output = Dict(
      "metadata" => Dict(
        "timestamp" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        "neuron_count" => 100,
        "maxiters" => maxiters,
        "milestone_interval" => milestone_interval,
        "optimizer" => "adam",
        "seed" => 1234,
        "N" => settings.N,
        "num_supervised" => settings.num_supervised,
        "domain" => [Float64(settings.x_left), Float64(settings.x_right)],
        "weights" => Dict(
          "supervised" => Float64(settings.supervised_weight),
          "bc" => Float64(settings.bc_weight),
          "pde" => Float64(settings.pde_weight)
        ),
        "ode_matrix" => ode_matrix,
        "benchmark_coefficients" => benchmark_coefficients
      ),
      "final" => Dict(
        "objective" => Float64(final_error),
        "coefficients" => final_coeffs
      ),
      "milestones" => [
        Dict(
          "iteration" => m.iteration,
          "objective" => m.objective,
          "coefficients" => m.coefficients
        ) for m in milestones
      ]
    )

    results_file = joinpath(output_dir, "scaling_adam_results.json")
    open(results_file, "w") do io
      JSON.print(io, output, 2)
    end

    # Loss CSV is already written by train_pinn

    @info "Training complete for example $run_idx"
    @info "Results written to $results_file"
  end
end


# Essentially the code is the same but it is just for LBFGs now. I do not think we will keep this... may have to delete it
# because I am just commenting out the adam training part.
function scaling_lbfgs(settings::TrainingSchemesSettings, iteration_counts::Dict{String,Int})
  array_of_benchmark_loss = Float64[]
  for (filename, iteration_count) in iteration_counts
    @info "Starting training for $filename for $iteration_count"
    for (run_idx, inner_dict) in settings.training_dataset
      converted_dict = convert_plugboard_keys(inner_dict)

      pinn_settings = PINNSettings(100, 1234, converted_dict, iteration_count, settings.num_supervised, settings.N, settings.num_points, settings.x_left, settings.x_right, settings.supervised_weight, settings.bc_weight, settings.pde_weight, settings.xs, "lbfgs")

      run_id = generate_run_id(pinn_settings.optimizer)
      output_dir = joinpath("results", "run-$run_id")
      mkpath(output_dir)

      # Train the network
      p_trained, coeff_net, st, _, _ = train_pinn(pinn_settings, output_dir; run_id=run_id)
      function_error, _ = evaluate_solution(pinn_settings, p_trained, coeff_net, st, settings.benchmark_dataset["01"], output_dir, run_id)

      push!(array_of_benchmark_loss, function_error)
      @info "Function error: $function_error"
    end
  end
  # Save cross-run summary to results/ root
  mkpath("results")
  df = DataFrame(
    index = 1:length(array_of_benchmark_loss),
    function_error = array_of_benchmark_loss
  )
  CSV.write("./results/benchmark_losses.csv", df)
end



# Envokes the grid_search with increasing neuron count
function grid_search_at_scale(settings::TrainingSchemesSettings, neurons_counts::Dict{String,Int})
  for (filename, neuron_count) in neurons_counts
    run_id = generate_run_id("grid")
    output_dir = joinpath("results", "run-$run_id")
    mkpath(output_dir)

    println("Starting grid search with $neuron_count neurons → $output_dir")
    result = grid_search_2d(
      neuron_count,
      settings.training_dataset,
      settings.benchmark_dataset,
      :pde, (0.1, 1.0),  # supervised weight range
      :supervised, (0.1, 1.0),           # bc weight range
      100,                          # nxn grid search
      fixed_weights=(bc=1.0,),
      num_supervised=10, # num_supervised N output of coefficients
      N=10,
      x_left=0.0f0,
      x_right=1.0f0,
      xs=settings.xs,
      base_data_dir=output_dir
    )
  end
end



export TrainingSchemesSettings, scaling_neurons, grid_search_at_scale, scaling_adam, load_and_infer, replay_snapshots

end
