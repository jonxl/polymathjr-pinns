#=

This is the general script for the PINN.
This will be agnostic to architecture and size of the neural network (right now it is only for feedforeward)

Instead of the neural network approximating the solution u(x) directly, it learns
the optimal coefficients of a truncated power series that solves the ODE.

The process involves:
1. Defining the ODE and its boundary conditions.
2. Setting up a neural network that outputs a vector of coefficients.
3. Creating a loss function that measures how poorly the power series (built from the NN's coefficients)
   satisfies the ODE and boundary conditions.
4. Using an optimization algorithm (Adam) to train the network's parameters to minimize this loss.
5. Plotting the results to see how well our solution approximates the true, analytic solution.

=#

# ---------------------------------------------------------------------------
# Step 1: Import necessary libraries
# ---------------------------------------------------------------------------

module PINN

using Lux, ModelingToolkit
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote
using ComponentArrays
import IntervalSets: Interval
using Plots, ProgressMeter
import Random
using TaylorSeries
using CSV
using DataFrames
using JSON

DEBUG = false  # Toggle verbose debug output

using CUDA

include("../utils/gpu_utils.jl")
using .GPUUtils

include("../utils/ProgressBar.jl")
using .ProgressBar

include("../utils/loss_functions.jl")
using .loss_functions

include("../utils/helper_funcs.jl")
using .helper_funcs

# Ensure a "data" directory exists for saving plots.
isdir("data") || mkpath("data")

# Define the floating point type to use throughout the script (e.g., Float32).
# Using Float32 is standard for neural networks as it's computationally faster.
F = Float32

# ---------------------------------------------------------------------------
# Step 2: Define the PINN Settings Structure
# ---------------------------------------------------------------------------

struct PINNSettings
  neuron_num::Int
  seed::Int
  ode_matrices::Dict{Any,Any} # from the specific training run that is specified by the run number
  # maxiters_adam::Int
  maxiters_lbfgs::Int # no LBFGS
  n_terms_for_power_series::Int # The degree of the highest power term in the series.
  num_supervised::Int # The number of coefficients we will supervise during training.
  num_points::Int # number of points evaluated
  x_left::Float32 # left boundary 
  x_right::Float32 # right boundary 
  supervised_weight::Float32
  bc_weight::Float32 # for now we are going to test the two of these to zero
  pde_weight::Float32
  xs::Any
  optimizer::String
end

# We need to have a parameter for the PINN to allow us to swap architectures easily
# ---------------------------------------------------------------------------
# Step 3: Define the Mathematical Problem (The ODE)
# ---------------------------------------------------------------------------

# Using ModelingToolkit, we define the independent variable `x` and the dependent variable `u(x)`.
@parameters x
@variables u(..)

# Differential operator
Dx = Differential(x)

# Define the ordinary differential equation: u'(x) + u(x) = 0
# equation = Dx(u(x)) + u(x) ~ 0

# Define the domain over which the ODE is valid.
# domains = [x ∈ Interval(x_left, x_right)]

# ---------------------------------------------------------------------------
# Step 4: Setup the Power Series and Neural Network
# ---------------------------------------------------------------------------

# We will approximate the solution u(x) with a truncated power series of degree N.
N = 21 # The degree of the highest power term in the series.

# Pre-calculate factorials (0!, 1!, ..., N!) for use in the series.

num_supervised = 21 # The number of coefficients we will supervise during training.

# Create a set of points inside the domain to enforce the ODE. These are called "collocation points".
num_points = 10


# Domain boundaries
x_left = F(0.0)  # Left boundary of the domain
x_right = F(1.0) # Right boundary of the domain

# Define a weight for the boundary condition, surpivesed coefficients, and the pde
# supervised_weight = F(1.0)  # Weight for the supervised loss term in the total loss function.
# bc_weight = F(1.0) # for now we are going to test the two of these to zero
# pde_weight = F(1.0)

# ---------------------------------------------------------------------------
# Step 5: Initialize Neural Network with Settings
# ---------------------------------------------------------------------------

function initialize_network(settings::PINNSettings; use_gpu::Bool=false)
  # Find the maximum matrix dimensions for input layer size
  max_input_size = if !isempty(settings.ode_matrices)
    maximum(prod(size(key)) for (key, _) in settings.ode_matrices)
  else
    settings.n_terms_for_power_series + 1
  end

  coeff_net = Lux.Chain(
    Lux.Dense(max_input_size, settings.neuron_num, σ),
    Lux.Dense(settings.neuron_num, settings.neuron_num, σ),
    Lux.Dense(settings.neuron_num, settings.neuron_num, σ),
    Lux.Dense(settings.neuron_num, settings.n_terms_for_power_series + 1)
  )

  # Initialize the network's parameters with the specified seed
  rng = Random.default_rng()
  Random.seed!(rng, settings.seed)

  p_init, st = Lux.setup(rng, coeff_net)

  # Wrap the initial parameters in a ComponentArray
  p_init_ca = ComponentArray(p_init)

  # Transfer parameters to GPU if available
  if use_gpu
    p_init_ca = CUDA.cu(p_init_ca)
    @info "Network parameters transferred to GPU"
  end

  return coeff_net, p_init_ca, st
end

# ---------------------------------------------------------------------------
# Step 6: Define the Loss Function
# ---------------------------------------------------------------------------

function loss_fn(p_net, data, coeff_net, st, ode_matrix_flat, boundary_condition, settings::PINNSettings, use_gpu::Bool=false; ode_buffers::Union{ODEBuffers, Nothing}=nothing)
  # Use pre-computed device arrays when available; otherwise transfer on each call (CPU fallback)
  ode_flat_dev, bc_dev, data_dev, xs_dev = if ode_buffers !== nothing
    (ode_buffers.ode_flat_dev, ode_buffers.bc_dev, ode_buffers.data_dev, ode_buffers.xs_dev)
  else
    Zygote.ignore() do
      (GPUUtils.to_device(ode_matrix_flat; gpu=use_gpu),
       GPUUtils.to_device(boundary_condition; gpu=use_gpu),
       GPUUtils.to_device(data; gpu=use_gpu),
       GPUUtils.to_device(collect(settings.xs); gpu=use_gpu))
    end
  end

  # Run the network to get coefficients (output is on same device as p_net)
  a_vec = vec(first(coeff_net(ode_flat_dev, p_net, st)))

  loss_func_settings = LossFunctionSettings(
    a_vec,
    settings.n_terms_for_power_series,
    ode_flat_dev,
    settings.x_left,
    bc_dev,
    xs_dev,
    settings.num_points,
    settings.num_supervised,
    data_dev,
  )

  # Calculate the PDE loss (residual of the ODE)
  loss_pde = generate_loss_pde_value(loss_func_settings; ode_buffers=ode_buffers)

  # Calculate the loss from the boundary conditions
  loss_bc = generate_loss_bc_value(loss_func_settings; ode_buffers=ode_buffers)

  # Calculate supervised loss using the plugboard coefficients
  loss_supervised = generate_loss_supervised_value(loss_func_settings; ode_buffers=ode_buffers)

  # The total loss is a weighted sum of the components
  return loss_pde * settings.pde_weight + settings.bc_weight * loss_bc + settings.supervised_weight * loss_supervised, loss_bc, loss_pde, loss_supervised
end

# ---------------------------------------------------------------------------
# Step 7: Global Loss Function
# ---------------------------------------------------------------------------

function global_loss(p_net, settings::PINNSettings, coeff_net, st, use_gpu::Bool=false; all_ode_buffers::Union{Dict, Nothing}=nothing)
  total_loss = F(0.0)
  total_local_loss_bc = F(0.0)
  total_local_loss_pde = F(0.0)
  total_local_loss_supervised = F(0.0)
  num_of_training_examples = length(settings.ode_matrices)

  for (alpha_matrix_key, series_coeffs) in settings.ode_matrices
    matrix_flat = vec(alpha_matrix_key)
    boundary_condition = [series_coeffs[1], series_coeffs[2]]
    # Look up pre-computed buffers for this ODE (nothing if not available)
    buffers = all_ode_buffers !== nothing ? get(all_ode_buffers, alpha_matrix_key, nothing) : nothing
    local_loss, local_loss_bc, local_loss_pde, local_loss_supervised = loss_fn(p_net, series_coeffs, coeff_net, st, matrix_flat, boundary_condition, settings, use_gpu; ode_buffers=buffers)
    total_loss += local_loss
    total_local_loss_bc += local_loss_bc
    total_local_loss_pde += local_loss_pde
    total_local_loss_supervised += local_loss_supervised
  end

  normalized_loss = total_loss / num_of_training_examples

  # Breakdown is for logging only — keep it off the AD tape
  losses = Zygote.ignore() do
    (
      bc  = Float32(total_local_loss_bc / num_of_training_examples),
      pde = Float32(total_local_loss_pde / num_of_training_examples),
      sup = Float32(total_local_loss_supervised / num_of_training_examples)
    )
  end

  return normalized_loss, losses
end

# ---------------------------------------------------------------------------
# Step 8: Training Function
# ---------------------------------------------------------------------------

#=
We train the PINN on the training dataset and return the network
=#

function train_pinn(settings::PINNSettings, output_dir; run_id::String=generate_run_id(settings.optimizer), milestone_interval::Int=0, on_milestone::Union{Function,Nothing}=nothing, progress_callback::Union{Function,Nothing}=nothing, write_loss_csv::Bool=true, snapshot_path::Union{String,Nothing}=nothing)
  csv_file = joinpath(output_dir, "loss.csv")

  use_gpu = GPUUtils.is_gpu_available()

  # Only log device info when using own progress bar (not in grid search)
  if progress_callback === nothing
    if use_gpu
      @info "Training on GPU: $(GPUUtils.get_device())"
    else
      @info "Training on CPU"
    end
  end

  coeff_net, p_init_ca, st = initialize_network(settings; use_gpu=use_gpu)

  # Warm-start from snapshot if provided
  if snapshot_path !== nothing
    raw = reinterpret(Float32, read(snapshot_path))
    p_init_ca = ComponentArray(raw, getaxes(p_init_ca))
    if use_gpu
      p_init_ca = CUDA.cu(p_init_ca)
    end
    @info "Loaded weights from snapshot: $snapshot_path"
  end

  # Pre-compute all constant ODE data on the target device (GPU or CPU) once
  to_device_fn = x -> GPUUtils.to_device(x; gpu=use_gpu)
  all_ode_buffers = precompute_buffers(settings, use_gpu, to_device_fn)
  @info "Pre-computed ODE buffers for $(length(all_ode_buffers)) training examples (device: $(use_gpu ? "GPU" : "CPU"))"

  # How often to record metrics and update the progress bar (every N iterations)
  LOG_INTERVAL = 100

  # Pre-allocate history buffer: columns = [total, bc, pde, supervised], rows = sampled iterations
  max_logged = cld(settings.maxiters_lbfgs, LOG_INTERVAL) + 1  # ceiling division + 1 for safety
  history_buf = Matrix{Float32}(undef, max_logged, 4)
  history_iters = Vector{Int}(undef, max_logged)
  history_len = Ref(0)

  latest_metrics = Ref((0.0f0, 0.0f0, 0.0f0))
  latest_params = Ref{Any}(p_init_ca)  # track latest params for graceful interrupt
  iter_count = Ref(0)

  # Create wrapper function for optimization — captures pre-computed buffers
  function loss_wrapper(p_net, _)
    loss, losses = global_loss(p_net, settings, coeff_net, st, use_gpu; all_ode_buffers=all_ode_buffers)
    latest_metrics[] = (losses.bc, losses.pde, losses.sup)
    return loss
  end

  function custom_callback(state, l; progress_bar)
    iter_count[] += 1
    iteration = iter_count[]
    latest_params[] = state.u

    # Only record metrics + update progress bar every LOG_INTERVAL iterations
    if iteration % LOG_INTERVAL == 0 || iteration == 1
      bc, pde, sup = latest_metrics[]
      idx = history_len[] + 1
      history_len[] = idx
      history_iters[idx] = iteration
      history_buf[idx, 1] = Float32(l)
      history_buf[idx, 2] = Float32(bc)
      history_buf[idx, 3] = Float32(pde)
      history_buf[idx, 4] = Float32(sup)
      progress_bar(state, l)
    end

    # Check if we've hit a milestone
    if on_milestone !== nothing && milestone_interval > 0 && iteration % milestone_interval == 0
      p_current = use_gpu ? ComponentArray(Array(getdata(state.u)), getaxes(state.u)) : state.u
      on_milestone(p_current, iteration, coeff_net, st, run_id)
    end

    return false
  end

  adtype = Optimization.AutoZygote()
  optfun = OptimizationFunction(loss_wrapper, adtype)
  prob = OptimizationProblem(optfun, p_init_ca)

  # ---------------- Adam ----------------
  if progress_callback !== nothing
    # External callback provided (e.g. shared grid search progress bar)
    callback_bar = progress_callback
  else
    # Default: create own progress bar (single run / scaling adam)
    @info "Starting Adam optimization..."
    p_bar = ProgressBar.ProgressBarSettings(settings.maxiters_lbfgs, "Adam...")
    callback_bar = ProgressBar.Bar(p_bar; step_size=LOG_INTERVAL)
  end

  adam_opt = OptimizationOptimisers.Adam(0.001f0)

  interrupted = false
  res = try
    solve(prob,
      adam_opt;
      callback = (state, l) -> custom_callback(state, l; progress_bar=callback_bar),
      maxiters=settings.maxiters_lbfgs)
  catch e
    if e isa InterruptException
      interrupted = true
      @warn "Training interrupted at iteration $(iter_count[]). Saving progress..."
      (u = latest_params[],)  # mock result with latest params
    else
      rethrow(e)
    end
  end

  #=
  # ---------------- LBFGS (disabled) ----------------
  @info "Starting LBFGS fine-tuning..."
  p_bar = ProgressBar.ProgressBarSettings(settings.maxiters_lbfgs, "LBFGS fine-tune...")
  callback_bar = ProgressBar.Bar(p_bar)

  lbfgs_opt = OptimizationOptimJL.LBFGS()

  res = solve(prob,
    lbfgs_opt;
    callback = (state, l) -> custom_callback(state, l; progress_bar=callback_bar),
    maxiters=settings.maxiters_lbfgs)
  =#

  # Extract final trained parameters — transfer back to CPU for evaluation/plotting
  p_trained = use_gpu ? ComponentArray(Array(getdata(res.u)), getaxes(res.u)) : res.u

  n = history_len[]
  if interrupted
    @info "Partial training saved — $(iter_count[]) iterations completed ($n logged samples)."
  else
    @info "Training complete — $(iter_count[]) iterations ($n logged samples)."
  end

  # Build history from pre-allocated buffer (only the filled portion)
  history = [(
    total = history_buf[i, 1],
    bc    = history_buf[i, 2],
    pde   = history_buf[i, 3],
    supervised = history_buf[i, 4]
  ) for i in 1:n]

  if write_loss_csv && n > 0
    df = DataFrame(
      iteration  = history_iters[1:n],
      total      = history_buf[1:n, 1],
      bc         = history_buf[1:n, 2],
      pde        = history_buf[1:n, 3],
      supervised = history_buf[1:n, 4]
    )
    CSV.write(csv_file, df)
  end

  return p_trained, coeff_net, st, run_id, history
end

# ---------------------------------------------------------------------------
# Step 9: Evaluation and Analysis Functions
# ---------------------------------------------------------------------------

# This code is the true solution for the ODE
# analytic_sol_func(x) = (pi * x * (-x + (pi^2) * (2x - 3) + 1) - sin(pi * x)) / (pi^3) # We replace with our training examples
# This is then represented as a TaylorSeries 

function evaluate_solution(settings::PINNSettings, p_trained, coeff_net, st, benchmark_dataset, output_dir, run_id; iteration::Int=0, write_results_json::Bool=true)
  converted_benchmark_dataset = convert_plugboard_keys(benchmark_dataset)
  fact = factorial.(big.(0:settings.n_terms_for_power_series))

  # We will update the error. For now we are only going to do ONE test set.
  loss = F(0.0)
  all_results = Dict[]

  for (alpha_matrix_key, benchmark_series_coeffs) in converted_benchmark_dataset
    matrix_flat = Float32.(vec(alpha_matrix_key))  # Flatten to Float32 column vector
    boundary_condition = [benchmark_series_coeffs[1], benchmark_series_coeffs[2]]

    benchmark_loss, _, _, _ = loss_fn(p_trained, benchmark_series_coeffs, coeff_net, st, matrix_flat, boundary_condition, settings::PINNSettings)
    loss += benchmark_loss

    a_learned = first(coeff_net(matrix_flat, p_trained, st))[:, 1] # extract learned coefficients

    # Write results.json — self-contained output for nn-viewer
    results = Dict(
      "alpha_matrix" => vec(alpha_matrix_key),
      "benchmark_coefficients" => benchmark_series_coeffs,
      "pinn_coefficients" => Float64.(a_learned),
      "function_error" => Float64(loss),
      "iteration" => iteration
    )
    push!(all_results, results)

    if write_results_json
      results_file = joinpath(output_dir, "results.json")
      entry_id = generate_run_id(settings.optimizer)
      append_to_results_json(results_file, entry_id, results)

      @info "Results written to $results_file (run: $run_id)"
    end
    if DEBUG
      @info "PINN's guess for coefficients: $a_learned"
      @info "The REAL coefficients: $benchmark_series_coeffs"
    end
  end

  ### ========================================================================
  ### IMPLEMENTATION BEFORE NN-VIEWER
  ### https://github.com/jonxlegasa/nn-viewer
  ### All graph generation below is disabled — visualization is now handled
  ### by the nn-viewer UI library which reads results.json
  ### ========================================================================
  #=
  for (alpha_matrix_key, benchmark_series_coeffs) in converted_benchmark_dataset
    matrix_flat = Float32.(vec(alpha_matrix_key))
    a_learned = first(coeff_net(matrix_flat, p_trained, st))[:, 1]

    # NOTE: THIS WILL STILL BE USED, I JUST HAVE TO FIGUERE OUT WHERE
    # u_real_func(x) = sum(benchmark_series_coeffs[i] * x^(i - 1) / fact[i] for i in 1:settings.n_terms_for_power_series)

    # ODE Matrix [1; 6; 2;;]
    roots = quadratic_formula(1, 6, 2)
    c1 = (3 * roots[2] - 5) * (1/(roots[2] - roots[1]))
    c2 = (-3 * roots[1] + 5) * (1/(roots[2] - roots[1]))

    u_real_func(x) = c1 * exp(roots[1] * x) + c2 * exp(roots[2] * x)
    u_predict_func(x) = sum(a_learned[i] * x^(i - 1) / fact[i] for i in 1:settings.n_terms_for_power_series)

    x_plot = settings.x_left:F(0.01):settings.x_right
    u_real = u_real_func.(x_plot)
    u_predict = u_predict_func.(x_plot)

    # FIGURE 1: Function Analysis (u(x) comparison and error)
    function_comparison = plot(x_plot, u_real,
      label="Analytic Solution", linestyle=:dash, linewidth=3,
      title="ODE Solution Comparison", xlabel="x", ylabel="u(x)",
      yscale=:log10, legend=:best)
    plot!(function_comparison, x_plot, u_predict, label="PINN Power Series", linewidth=2)

    function_error_data = max.(abs.(u_real .- u_predict), F(1e-20))
    function_error_plot = plot(x_plot, function_error_data,
      title="Absolute Error of Solution", label="|Analytic - Predicted|",
      yscale=:log10, xlabel="x", ylabel="Error", linewidth=2)

    figure_one = plot(function_comparison, function_error_plot, layout=(2, 1), size=(800, 800))
    savefig(figure_one, data_directories[1])

    # FIGURE 2: Coefficient Analysis (comparison and error)
    n_length_benchmark = length(benchmark_series_coeffs)
    indices = 1:n_length_benchmark

    coefficient_comparison = plot(indices, benchmark_series_coeffs,
      title="Coefficient Comparison", label="Benchmark",
      xlabel="Coefficient Index", ylabel="Coefficient Value", legend=:best)
    plot!(coefficient_comparison, indices, a_learned[1:n_length_benchmark], label="PINN")

    coefficient_error_data = max.(abs.(benchmark_series_coeffs .- a_learned[1:n_length_benchmark]), 1e-20)
    coefficient_error_plot = plot(indices, coefficient_error_data,
      title="Absolute Error of Coefficients", label="|Benchmark - PINN|",
      yscale=:log10, xlabel="Coefficient Index", ylabel="Absolute Error", linewidth=2)

    figure_two = plot(coefficient_comparison, coefficient_error_plot, layout=(2, 1), size=(800, 800))
    savefig(figure_two, data_directories[2])

    # FIGURE 3: Loss iteration plots
    df = CSV.read(data_directories[6], DataFrame)

    function get_loss_values(df, loss_type_name)
      row = df[df.loss_type.==loss_type_name, :]
      if nrow(row) == 0
        return Float32[]
      end
      return Vector{Float32}(collect(skipmissing(row[1, 2:end])))
    end

    total_loss = get_loss_values(df, "total_loss")
    total_loss_bc = get_loss_values(df, "total_loss_bc")
    total_loss_pde = get_loss_values(df, "total_loss_pde")
    total_loss_supervised = get_loss_values(df, "total_loss_supervised")

    total_loss_plot = plot(1:length(total_loss), total_loss,
      title="Global Loss per Global Loss Call", xlabel="Loss Call", ylabel="Global Loss", yscale=:log10)
    total_bc_loss_plot = plot(1:length(total_loss_bc), total_loss_bc,
      title="Global BC Loss per Global Loss Call", xlabel="Loss Call", ylabel="BC Loss", yscale=:log10)
    total_pde_loss_plot = plot(1:length(total_loss_pde), total_loss_pde,
      title="Global PDE Loss per Global Loss Call", xlabel="Loss Call", ylabel="PDE Loss", yscale=:log10)
    total_supervised_loss_plot = plot(1:length(total_loss_supervised), total_loss_supervised,
      title="Global Supervised Loss per Global Loss Call", xlabel="Loss Call", ylabel="Supervised Loss", yscale=:log10)

    iteration_plot = plot(total_loss_plot, total_bc_loss_plot, total_pde_loss_plot, total_supervised_loss_plot,
      layout=(4, 1), size=(1000, 1000))
    savefig(iteration_plot, data_directories[5])
  end
  =#

  return loss, all_results
end

# ---------------------------------------------------------------------------
# Step 10: Export Functions
# ---------------------------------------------------------------------------

export PINNSettings, train_pinn, global_loss, evaluate_solution, initialize_network

end
