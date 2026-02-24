using Dates
using JSON
using CSV
using DataFrames

# Include util functions
include("../utils/plugboard.jl")
using .Plugboard

include("../utils/ProgressBar.jl")
using .ProgressBar

include("../utils/helper_funcs.jl")
using .helper_funcs

include("../utils/two_d_grid_search_hyperparameters.jl")
using .TwoDGridSearchOnWeights

include("../modelcode/PINN.jl")
using .PINN

include("../utils/training_schemes.jl")
using .training_schemes

# Dataset file paths
training_data_dir = "./data/training_dataset.json"
benchmark_data_dir = "./data/benchmark_dataset.json"

# =========================================================================
# Configuration: Data Generation
# =========================================================================
# Controls whether and how training/benchmark datasets are created.

GENERATE_DATASET = true   # true = regenerate datasets via plugboard, false = use existing JSON files
MODE = "RANDOM"          # "SPECIFIC" = hardcoded test_matrix, "RANDOM" = random ODE matrices
# =========================================================================
# Configuration: Training Mode
# =========================================================================
# Controls which training strategy to run. Set ONE of these to true.
#   SINGLE_RUN   — Train one PINN with fixed hyperparameters
#   GRID_SEARCH  — Threaded 2D grid search over loss weights (pde x supervised)
#   SCALING_ADAM — Iterate over training dataset with milestone evaluation

TRAINING_MODE = "SCALING_ADAM"  # "SINGLE_RUN", "GRID_SEARCH", or "SCALING_ADAM"

# =========================================================================
# Configuration: Snapshots
# =========================================================================
# Saving: periodically write model weights to results/run-{id}/snapshots/
# Loading: warm-start training from a previously saved snapshot

SAVE_SNAPSHOTS = true   # true = save weight snapshots during training, false = skip
SNAPSHOT_INTERVAL = 100  # save a snapshot every N iterations (only when SAVE_SNAPSHOTS = true)

LOAD_SNAPSHOT = false   # true = warm-start from snapshot, false = train from scratch
SNAPSHOT_PATH = ""      # path to .bin snapshot file, e.g. "results/run-adam-a8Kf3x2Q/snapshots/iter-0100000.bin"

#=
This function does the following:
Create training run directories
=#
function create_training_run_dirs(run_number::Int64, batch_size::Any)
  """
  Creates a training run directory and output file with specified naming convention.
  Args:
      run_number: The training run number (will be zero-padded to 2 digits)
      training_examples: Array of natural numbers representing training examples for each model
  """
  # Create data directory if it doesn't exist
  data_dir = "data"
  if !isdir(data_dir)
    mkdir(data_dir)
    @info "Created data directory: $data_dir"
  end

  # Format run number with zero padding (01, 02, 03, etc.)
  run_number_formatted = lpad(run_number, 2, '0')
  # Create training run directory
  training_run_dir = joinpath(data_dir, "training-run-$run_number_formatted")
  if !isdir(training_run_dir)
    mkdir(training_run_dir)
    @info "Created training run directory: $training_run_dir"
  end

  # Generate output file with training run information
  output_file = joinpath(training_run_dir, "training_info.txt")
  # Get current date and time
  current_datetime = now()

  # Write training run information to file
  open(output_file, "w") do file
    println(file, "Training Run Information")
    println(file, "="^30)
    println(file, "Training Run Number: $run_number_formatted")
    println(file, "Training Examples per Model: $batch_size")
    println(file, "Training Run Commenced: $current_datetime")
    println(file, "="^30)
  end

  @info "Training run $run_number_formatted setup complete" output_file

  return training_run_dir, output_file
end


#=
This function initializes training run batches
and creates training and benchmark dataset
=#

function init_batches(batch_sizes::Array{Int})
  """
  Initializes batches by generating datasets for different batch sizes.
  Args:
      batch_sizes: Array of integers representing different batch sizes
  """

  benchmark_dataset_setting::Settings = Plugboard.Settings(1, 0, 1, benchmark_data_dir, 10)

  # generate training datasets and benchmarks 
  for (batch_index, k) in enumerate(batch_sizes)
    training_dataset_setting::Settings = Plugboard.Settings(1, 0, k, training_data_dir, 10)
    # set up plugboard for solutions to ay' + by = 0 where a,b != 0
    run_number_formatted = lpad(batch_index, 2, '0')

    @info "Generating datasets for batch $run_number_formatted" num_examples=k mode=MODE

    # Training data depends on MODE
    specific_training_matrix = [-1; 1;;]
    if MODE == "SPECIFIC"
      @warn "In $MODE mode. Generating specific training dataset for $specific_matrix"
      Plugboard.generate_specific_ode_dataset(training_dataset_setting, batch_index, specific_training_matrix)
    elseif MODE == "RANDOM"
      Plugboard.generate_random_ode_dataset(training_dataset_setting, batch_index)
    end

    # specific_benchmark_matrix = [1; 1;;]

    Plugboard.generate_random_ode_dataset(benchmark_dataset_setting, 1)

    # Benchmark always uses the specific test matrix for consistent evaluation
    # Plugboard.generate_specific_ode_dataset(benchmark_dataset_setting, 1, specific_benchmark_matrix)
  end
end

#= 
This function takes the batches and their sizes and runs
the PINN on the training dataset
=#

function run_training_sequence(batch_sizes::Array{Int})
  """
  Runs a sequence of training runs with different training example configurations.
  Args:
      batch_sizes: Array of integers representing different batch sizes
  """
  # Initialize all batches first (generate datasets via plugboard)
  if GENERATE_DATASET
    init_batches(batch_sizes)
  end

  # we only load the training data dir here
  training_dataset = JSON.parsefile(training_data_dir)
  benchmark_dataset = JSON.parsefile(benchmark_data_dir)

  F = Float32
  # We will approximate the solution u(x) with a truncated power series of degree N.
  # BS on pde_weight with supervised and bc fixed at 1.0

  N = 10 # The degree of the highest power term in the series.

  num_supervised = 10 # The number of coefficients we will supervise during training.
  # Create a set of points inside the domain to enforce the ODE. These are called "collocation points".
  num_points = 10

  # Domain boundaries
  x_left = F(0.0)  # Left boundary of the domain
  x_right = F(1.0) # Right boundary of the domain

  # Define a weight for the boundary condition, supervised coefficients, and the pde
  supervised_weight = F(1.0)
  bc_weight = F(1.0)
  pde_weight = F(1.0)

  xs = range(x_left, x_right, length=num_points)

  # Ensure parent results directory exists
  mkpath("results")

  # ---- Run selected training mode ----
  if TRAINING_MODE == "SINGLE_RUN"
    # Train one PINN with fixed hyperparameters
    for (run_idx, inner_dict) in training_dataset
      converted_dict = convert_plugboard_keys(inner_dict)

      float_converted_dict = Dict{Any, Any}()
      for (mat, series) in converted_dict
        float_converted_dict[Float32.(mat)] = series
      end

      settings = PINNSettings(10, 1234, float_converted_dict, 1000, N, num_supervised, num_points, x_left, x_right, supervised_weight, bc_weight, pde_weight, xs, "adam")

      run_id = generate_run_id(settings.optimizer)
      output_dir = joinpath("results", "run-$run_id")
      mkpath(output_dir)

      snap = LOAD_SNAPSHOT ? SNAPSHOT_PATH : nothing
      p_trained, coeff_net, st, _, _ = train_pinn(settings, output_dir; run_id=run_id, snapshot_path=snap)
      function_error, _ = evaluate_solution(settings, p_trained, coeff_net, st, benchmark_dataset["01"], output_dir, run_id)
      @info "Function error: $function_error"
    end

  elseif TRAINING_MODE == "GRID_SEARCH"
    # Threaded 2D grid search over pde_weight x supervised_weight
    # Launch with: julia --project -t auto src/main.jl
    neuron_count = 20
    run_id = generate_run_id("grid")
    output_dir = joinpath("results", "run-$run_id")
    mkpath(output_dir)

    result = grid_search_2d(
      neuron_count,
      training_dataset,
      benchmark_dataset,
      :pde, (0.1, 1.0),
      :supervised, (0.1, 1.0),
      2;
      fixed_weights=(bc=1.0,),
      num_supervised=num_supervised,
      N=N,
      x_left=x_left,
      x_right=x_right,
      xs=xs,
      base_data_dir=output_dir
    )
    @info "Grid search complete" best_objective=result.best_objective best_weights=result.best_weights

  elseif TRAINING_MODE == "SCALING_ADAM"
    # Iterate over training dataset with milestone callbacks
    maxiters = 1000
    milestone_interval = SAVE_SNAPSHOTS ? SNAPSHOT_INTERVAL : 0

    scaling_adam_settings = TrainingSchemesSettings(training_dataset, benchmark_dataset, N, num_supervised, num_points, x_left, x_right, supervised_weight, bc_weight, pde_weight, xs)
    snap = LOAD_SNAPSHOT ? SNAPSHOT_PATH : nothing
    scaling_adam(scaling_adam_settings, maxiters, milestone_interval; snapshot_path=snap)

  else
    error("Unknown TRAINING_MODE: $TRAINING_MODE. Use \"SINGLE_RUN\", \"GRID_SEARCH\", or \"SCALING_ADAM\".")
  end
end

batch = [1000]

run_training_sequence(batch)
