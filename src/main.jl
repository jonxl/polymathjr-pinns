using ArgParse
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

include("../architectures/PINN.jl")
using .PINN

include("../utils/training_schemes.jl")
using .training_schemes

# Dataset file paths
training_data_dir = "./data/training_dataset.json"
benchmark_data_dir = "./data/benchmark_dataset.json"

# =========================================================================
# CLI Argument Parsing
# =========================================================================
function parse_commandline()
  s = ArgParseSettings(description="PINN training for ODEs via power series")

  @add_arg_table! s begin
    "--mode"
      help = "Training mode: TRAIN or GRID_SEARCH"
      arg_type = String
      default = "TRAIN"
    "--gen-data"
      help = "Regenerate datasets via plugboard before training"
      action = :store_true
    "--data"
      help = "Dataset generation mode: RANDOM or SPECIFIC"
      arg_type = String
      default = "RANDOM"
    "--no-snap"
      help = "Disable saving weight snapshots during training"
      action = :store_true
    "--snap-every"
      help = "Legacy iteration checkpoint interval; mini-batch checkpoint cadence is controlled by --epochs"
      arg_type = Int
      default = 100
    "--resume"
      help = "Path to .safetensors snapshot file for warm-start (.bin legacy snapshots are still supported)"
      arg_type = String
      default = nothing
    "--bins"
      help = "ODEs per bin. 0 = full batch (all ODEs per iteration)"
      arg_type = Int
      default = 32
    "--epochs"
      help = "Save an intermediate checkpoint after every N complete epochs (mini-batch mode)"
      arg_type = Int
      default = 10
    "--maxiters"
      help = "Maximum number of training iterations (gradient updates)"
      arg_type = Int
      default = 10000
  end

  return parse_args(s)
end

parsed_args = parse_commandline()

# =========================================================================
# Configuration: From CLI args (override with command-line flags)
# =========================================================================
GENERATE_DATASET = parsed_args["gen-data"]
MODE = parsed_args["data"]
TRAINING_MODE = parsed_args["mode"]

SAVE_SNAPSHOTS = !parsed_args["no-snap"]
SNAPSHOT_INTERVAL = parsed_args["snap-every"]

LOAD_SNAPSHOT = parsed_args["resume"] !== nothing
SNAPSHOT_PATH = something(parsed_args["resume"], "")

BIN_SIZE = parsed_args["bins"]
SNAPSHOT_EVERY_N_EPOCHS = parsed_args["epochs"]
MAXITERS = parsed_args["maxiters"]

# =========================================================================
# Configuration: PINN Hyperparameters (in-file constants)
# =========================================================================
NEURON_COUNT = 100
SEED = 1234
N = 10                        # Power series degree
NUM_SUPERVISED = 10           # Supervised coefficients
NUM_POINTS = 10               # Collocation points
X_LEFT = Float32(0.0)
X_RIGHT = Float32(1.0)
SUPERVISED_WEIGHT = Float32(1.0)
BC_WEIGHT = Float32(1.0)
PDE_WEIGHT = Float32(1.0)

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

  xs = range(X_LEFT, X_RIGHT, length=NUM_POINTS)

  # Ensure parent results directory exists
  mkpath("results")

  # ---- Run selected training mode ----
  if TRAINING_MODE == "TRAIN"
    milestone_interval = SAVE_SNAPSHOTS ? SNAPSHOT_INTERVAL : 0

    train_settings = TrainingSchemesSettings(
      training_dataset, benchmark_dataset,
      N, NUM_SUPERVISED, NUM_POINTS,
      X_LEFT, X_RIGHT,
      SUPERVISED_WEIGHT, BC_WEIGHT, PDE_WEIGHT, xs)

    snap = LOAD_SNAPSHOT ? SNAPSHOT_PATH : nothing
    run_training(train_settings, MAXITERS, milestone_interval;
                 snapshot_path=snap,
                 batch_size=BIN_SIZE,
                 snapshot_epoch_interval=SAVE_SNAPSHOTS ? SNAPSHOT_EVERY_N_EPOCHS : 0,
                 neuron_count=NEURON_COUNT,
                 seed=SEED)

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
      num_supervised=NUM_SUPERVISED,
      N=N,
      x_left=X_LEFT,
      x_right=X_RIGHT,
      xs=xs,
      base_data_dir=output_dir
    )
    @info "Grid search complete" best_objective=result.best_objective best_weights=result.best_weights

  else
    error("Unknown TRAINING_MODE: $TRAINING_MODE. Use \"TRAIN\" or \"GRID_SEARCH\".")
  end
end

batch = [1000]

run_training_sequence(batch)
