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

include("../utils/safetensors_utils.jl")
using .SafeTensorSnapshots

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
  pde_weight::Float32
  xs::Any
  optimizer::String
  # Which solution representation the network learns. Determines the input/output
  # layer widths (see io_dims), which loss triple is used, and how u(x) is
  # reconstructed for evaluation and plotting.
  #   :power_series — outputs ψ₀…ψ_N,  u(x) = Σ ψₙ xⁿ            (monomial basis)
  #   :eigenvalue   — outputs (μ,k,A,B), u(x) = e^{μx}[A·C(k,x) + B·S(k,x)]
  representation::Symbol
  # Network input encoding. `:coefficients` preserves the general flattened
  # ODE input; `:trace_determinant` gives both dual-study models the same
  # two-value (tau, delta) input while the loss still uses [delta, -tau, 1].
  input_encoding::Symbol
end

# Backwards-compatible constructor: existing positional call sites
# predate the representation field and all mean :power_series.
function PINNSettings(neuron_num, seed, ode_matrices, maxiters_lbfgs,
                      n_terms_for_power_series, num_supervised, num_points,
                      x_left, x_right, supervised_weight, pde_weight,
                      xs, optimizer)
  return PINNSettings(neuron_num, seed, ode_matrices, maxiters_lbfgs,
                      n_terms_for_power_series, num_supervised, num_points,
                      x_left, x_right, supervised_weight, pde_weight,
                      xs, optimizer, :power_series, :coefficients)
end

function PINNSettings(neuron_num, seed, ode_matrices, maxiters_lbfgs,
                      n_terms_for_power_series, num_supervised, num_points,
                      x_left, x_right, supervised_weight, pde_weight,
                      xs, optimizer, representation)
  return PINNSettings(neuron_num, seed, ode_matrices, maxiters_lbfgs,
                      n_terms_for_power_series, num_supervised, num_points,
                      x_left, x_right, supervised_weight, pde_weight,
                      xs, optimizer, representation, :coefficients)
end

# ---------------------------------------------------------------------------
# Representation → network input/output widths
# ---------------------------------------------------------------------------
#
# The trunk (hidden layers) is identical across representations; only the first
# and last Dense layers differ. Keeping this in one place means adding a new
# representation touches io_dims and the loss functions, nothing else.

"""
    io_dims(settings::PINNSettings) → (input_width, output_width)

Input and output layer widths implied by `settings.representation`.
"""
io_dims(settings::PINNSettings) = io_dims(Val(settings.representation), settings)

# Power series: input is the flattened (canonicalized) ODE coefficient matrix,
# output is one coefficient per power x⁰…x^N.
function io_dims(::Val{:power_series}, settings::PINNSettings)
  settings.input_encoding === :trace_determinant && return (2, settings.n_terms_for_power_series + 1)
  in_width = if !isempty(settings.ode_matrices)
    maximum(prod(size(key)) for (key, _) in settings.ode_matrices)
  else
    settings.n_terms_for_power_series + 1
  end
  return (in_width, settings.n_terms_for_power_series + 1)
end

# Eigenvalue: input is the trace/determinant pair (τ, Δ) of y'' - τy' + Δy = 0,
# output is the four unified-form parameters (μ, k, A, B).
io_dims(::Val{:eigenvalue}, ::PINNSettings) = (2, 4)

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

# Active objective weights.
# supervised_weight = F(1.0)  # Weight for the supervised loss term in the total loss function.
# pde_weight = F(1.0)

# ---------------------------------------------------------------------------
# Step 5: Initialize Neural Network with Settings
# ---------------------------------------------------------------------------

function initialize_network(settings::PINNSettings; use_gpu::Bool=false)
  # Layer widths come from the representation; the hidden trunk is shared.
  in_width, out_width = io_dims(settings)

  coeff_net = Lux.Chain(
    Lux.Dense(in_width, settings.neuron_num, σ),
    Lux.Dense(settings.neuron_num, settings.neuron_num, σ),
    Lux.Dense(settings.neuron_num, settings.neuron_num, σ),
    Lux.Dense(settings.neuron_num, out_width)
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

function loss_fn(p_net, data, coeff_net, st, ode_matrix_flat, boundary_condition, settings::PINNSettings, use_gpu::Bool=false; ode_buffers::Union{ODEBuffers, EigBuffers, Nothing}=nothing)
  # Eigenvalue representation: the network consumes (τ, Δ) and emits (μ, k, A, B),
  # and all three loss terms are analytic. Nothing below this branch applies.
  if settings.representation === :eigenvalue
    ode_buffers isa EigBuffers || error(
      "representation :eigenvalue requires EigBuffers (build them with precompute_eig_buffers), got $(typeof(ode_buffers))"
    )
    input_dev = Zygote.ignore() do
      ode_buffers.input_dev
    end
    out = vec(first(coeff_net(input_dev, p_net, st)))

    loss_pde = generate_loss_pde_value_eig(out, ode_buffers, settings.num_points)
    loss_bc = generate_loss_bc_value_eig(out, ode_buffers)
    loss_supervised = generate_loss_supervised_value_eig(out, ode_buffers)

    total = loss_pde * settings.pde_weight * settings.num_supervised +
            settings.supervised_weight * loss_supervised
    return total, loss_bc, loss_pde, loss_supervised
  end

  # Constants w.r.t. p_net — keep off AD tape whether pre-computed or transferred on the fly
  ode_flat_dev, bc_dev, data_dev, xs_dev = Zygote.ignore() do
    if ode_buffers !== nothing
      (ode_buffers.ode_flat_dev, ode_buffers.bc_dev, ode_buffers.data_dev, ode_buffers.xs_dev)
    else
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

  # Optimized objective excludes BC; BC is logged as a diagnostic.
  return loss_pde * settings.pde_weight * settings.num_supervised + settings.supervised_weight * loss_supervised, loss_bc, loss_pde, loss_supervised
end

# ---------------------------------------------------------------------------
# Step 6b: Epoch Batch Iterator (Mini-Batching)
# ---------------------------------------------------------------------------

mutable struct EpochBatchIterator
  all_items::Vector{Pair}    # all (matrix_key => series_coeffs) pairs
  order::Vector{Int}         # shuffled indices for current epoch
  pos::Int                   # current position in the epoch
  batch_size::Int            # ODEs per bin (0 = full batch)
  epoch_count::Int           # how many complete epochs have finished
  epoch_just_completed::Bool # true on the first call after an epoch wraps
end

function EpochBatchIterator(ode_matrices::Dict, batch_size::Int)
  items = collect(ode_matrices)
  order = batch_size > 0 ? Random.randperm(length(items)) : collect(1:length(items))
  EpochBatchIterator(items, order, 1, batch_size, 0, false)
end

function next_batch!(iter::EpochBatchIterator)
  iter.epoch_just_completed = false
  n = length(iter.all_items)
  if iter.batch_size <= 0 || iter.batch_size >= n
    # Full batch mode — every call is a complete "epoch"
    iter.epoch_count += 1
    iter.epoch_just_completed = true
    return iter.all_items
  end
  # If we've exhausted this epoch, re-shuffle and start a new one
  if iter.pos > n
    iter.epoch_count += 1
    iter.epoch_just_completed = true
    iter.order = Random.randperm(n)
    iter.pos = 1
  end
  # Take next chunk (may be smaller than batch_size for the last bin)
  batch_end = min(iter.pos + iter.batch_size - 1, n)
  indices = iter.order[iter.pos:batch_end]
  iter.pos = batch_end + 1
  return iter.all_items[indices]
end

# ---------------------------------------------------------------------------
# Step 7: Global Loss Function
# ---------------------------------------------------------------------------

"""
    global_loss_batched(p_net, settings, coeff_net, st, buf) → (loss, losses)

Batched equivalent of `global_loss`: one network call over the whole bin
instead of one per ODE. `buf` is a `BatchBuffers` (power series) or
`BatchEigBuffers` (eigenvalue), already sliced to the current bin.

Returns the same `(total, (bc=, pde=, sup=))` shape as `global_loss`, so the
training loop, loss CSV, and snapshot logic are unaffected by which path runs.

Verified numerically identical to the per-ODE path: forward within 1e-7
relative, gradients within 2e-7, at ~83x (power series) and ~31x (eigenvalue)
the speed for a 600-ODE bin.
"""
function global_loss_batched(p_net, settings::PINNSettings, coeff_net, st, buf)
  # One network call for the entire bin.
  out = first(coeff_net(buf.X, p_net, st))

  loss_pde, loss_bc, loss_sup = if buf isa BatchEigBuffers
    batched_eigenvalue_losses(out, buf)
  else
    batched_power_series_losses(out, buf)
  end

  total = loss_pde * settings.pde_weight * settings.num_supervised +
          loss_sup * settings.supervised_weight

  # Breakdown is for logging only — keep it off the AD tape
  losses = Zygote.ignore() do
    (bc = Float32(loss_bc), pde = Float32(loss_pde), sup = Float32(loss_sup))
  end

  return total, losses
end

function global_loss(p_net, settings::PINNSettings, coeff_net, st, use_gpu::Bool=false;
                     all_ode_buffers::Union{Dict, Nothing}=nothing,
                     ode_items::Union{Vector, Nothing}=nothing)
  # Use provided batch or fall back to full dataset
  items = ode_items !== nothing ? ode_items : collect(settings.ode_matrices)

  total_loss = F(0.0)
  total_local_loss_bc = F(0.0)
  total_local_loss_pde = F(0.0)
  total_local_loss_supervised = F(0.0)
  num_in_batch = length(items)

  for (alpha_matrix_key, series_coeffs) in items
    # Canonicalized so the fallback (no-buffers) path matches precompute_buffers
    matrix_flat = canonicalize_alpha(vec(alpha_matrix_key))
    boundary_condition = [series_coeffs[1], series_coeffs[2]]
    # Look up pre-computed buffers for this ODE — off the AD tape since buffers are constant w.r.t. p_net
    buffers = Zygote.ignore() do
      all_ode_buffers !== nothing ? get(all_ode_buffers, alpha_matrix_key, nothing) : nothing
    end
    local_loss, local_loss_bc, local_loss_pde, local_loss_supervised = loss_fn(p_net, series_coeffs, coeff_net, st, matrix_flat, boundary_condition, settings, use_gpu; ode_buffers=buffers)
    total_loss += local_loss
    total_local_loss_bc += local_loss_bc
    total_local_loss_pde += local_loss_pde
    total_local_loss_supervised += local_loss_supervised
  end

  normalized_loss = total_loss / num_in_batch

  # Breakdown is for logging only — keep it off the AD tape
  losses = Zygote.ignore() do
    (
      bc  = Float32(total_local_loss_bc / num_in_batch),
      pde = Float32(total_local_loss_pde / num_in_batch),
      sup = Float32(total_local_loss_supervised / num_in_batch)
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

function train_pinn(settings::PINNSettings, output_dir; run_id::String=generate_run_id(settings.optimizer), milestone_interval::Int=0, on_milestone::Union{Function,Nothing}=nothing, on_interrupt::Union{Function,Nothing}=nothing, progress_callback::Union{Function,Nothing}=nothing, write_loss_csv::Bool=true, snapshot_path::Union{String,Nothing}=nothing, batch_size::Int=0, snapshot_epoch_interval::Int=10, batch_provider::Union{Function,Nothing}=nothing, streaming_dataset_size::Int=0)
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

  if snapshot_path !== nothing
    coeff_net, p_ca, st, _ = SafeTensorSnapshots.load_any_model(snapshot_path)
    p_init_ca = use_gpu ? CUDA.cu(p_ca) : p_ca
    @info "Loaded model from snapshot: $snapshot_path"
  else
    coeff_net, p_init_ca, st = initialize_network(settings; use_gpu=use_gpu)
  end

  # Pre-compute all constant ODE data on the target device (GPU or CPU) once
  to_device_fn = x -> GPUUtils.to_device(x; gpu=use_gpu)
  # Batched path: build every ODE's columns ONCE for the whole dataset, then
  # slice per iteration with select_bin. The shared basis/derivative matrices
  # are bin-independent, so this is the only construction that ever happens.
  all_items = collect(settings.ode_matrices)
  streaming = batch_provider !== nothing
  streaming && streaming_dataset_size <= 0 && error("streaming_dataset_size must be positive when batch_provider is supplied")
  full_buffers = if streaming
    nothing
  elseif settings.representation === :eigenvalue
    precompute_batch_eig_buffers(settings, all_items, use_gpu, to_device_fn)
  else
    precompute_batch_buffers(settings, all_items, use_gpu, to_device_fn)
  end

  # Maps an ODE's matrix key to its column, so a bin of items becomes indices.
  key_to_col = Dict(k => i for (i, (k, _)) in enumerate(all_items))

  if streaming
    @info "Streaming $(settings.representation) batches from $(streaming_dataset_size) canonical training examples (device: $(use_gpu ? "GPU" : "CPU"))"
  else
    @info "Pre-computed batched $(settings.representation) buffers for $(length(all_items)) training examples (device: $(use_gpu ? "GPU" : "CPU"))"
  end

  # Create batch iterator for mini-batching (batch_size=0 means full batch)
  batch_iter = EpochBatchIterator(settings.ode_matrices, batch_size)
  n_odes = streaming ? streaming_dataset_size : length(settings.ode_matrices)
  batches_per_epoch = batch_size > 0 && batch_size < n_odes ? cld(n_odes, batch_size) : 1
  total_epochs = cld(settings.maxiters_lbfgs, batches_per_epoch)
  if batch_size > 0
    checkpoint_msg = snapshot_epoch_interval > 0 ? "checkpoints every $(snapshot_epoch_interval) epochs" : "intermediate checkpoints disabled"
    @info "Mini-batching enabled: $(n_odes) ODEs → $(batches_per_epoch) bins of ≤$(batch_size), $(checkpoint_msg)"
  end

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
  loss_call_count = Ref(0)

  # Create wrapper function for optimization — captures pre-computed buffers and batch iterator
  function loss_wrapper(p_net, _)
    # Bin selection and column slicing are constant w.r.t. p_net — keep both
    # off the AD tape. Slicing is cheap; the buffers themselves are never rebuilt.
    bin = Zygote.ignore() do
      if streaming
        loss_call_count[] += 1
        epoch = cld(loss_call_count[], batches_per_epoch)
        batch = mod1(loss_call_count[], batches_per_epoch)
        items = batch_provider(epoch, batch)
        settings.representation === :eigenvalue ?
          precompute_batch_eig_buffers(settings, items, use_gpu, to_device_fn) :
          precompute_batch_buffers(settings, items, use_gpu, to_device_fn)
      else
        items = next_batch!(batch_iter)
        if length(items) == length(all_items)
          full_buffers          # full batch — no slice needed
        else
          select_bin(full_buffers, [key_to_col[k] for (k, _) in items])
        end
      end
    end
    loss, losses = global_loss_batched(p_net, settings, coeff_net, st, bin)
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

    # Checkpoints are saved only at configured epoch boundaries. The final model
    # is saved separately by the training scheme after optimization completes.
    epoch_done = streaming ? iteration % batches_per_epoch == 0 : Zygote.ignore() do
      batch_iter.epoch_just_completed
    end
    completed_epoch = streaming ? iteration ÷ batches_per_epoch : batch_iter.epoch_count
    if on_milestone !== nothing && epoch_done &&
       snapshot_epoch_interval > 0 &&
       completed_epoch % snapshot_epoch_interval == 0
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
    callback_bar = ProgressBar.Bar(
      p_bar;
      step_size=LOG_INTERVAL,
      showvalues_fn=() -> [
        ("Iteration", "$(iter_count[]) / $(settings.maxiters_lbfgs)"),
        ("Epoch", "$(cld(iter_count[], batches_per_epoch)) / $(total_epochs)"),
      ],
    )
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
      it = iter_count[]
      p_current = use_gpu ? ComponentArray(Array(getdata(latest_params[])), getaxes(latest_params[])) : latest_params[]
      @warn "Training interrupted at iteration $it. Saving progress..."
      if on_interrupt !== nothing
        on_interrupt(p_current, it, coeff_net, st, run_id)
      end
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
  items = collect(converted_benchmark_dataset)
  buf = settings.representation === :eigenvalue ?
    precompute_batch_eig_buffers(settings, items, false, identity) :
    precompute_batch_buffers(settings, items, false, identity)
  out = first(coeff_net(buf.X, p_trained, st))
  U = batched_reconstruct(out, buf)
  UT = true_solutions(buf)
  pde, bc, sup = settings.representation === :eigenvalue ?
    batched_eigenvalue_losses(out, buf) :
    batched_power_series_losses(out, buf)

  loss = Float32(settings.pde_weight) * Float32(settings.num_supervised) * Float32(pde) +
         Float32(settings.supervised_weight) * Float32(sup)

  all_results = Dict[]

  for (idx, (alpha_matrix_key, benchmark_series_coeffs)) in enumerate(items)
    local_relerr = relative_l2(U[:, idx], UT[:, idx])
    learned = out[:, idx]

    result_payload = if settings.representation === :eigenvalue
      Dict{String,Any}(
        "representation_parameters" => Float64.(Vector(learned)),
        "parameter_names" => ["mu", "k", "A", "B"],
        "pinn_coefficients" => Float64.(Vector(learned)),
      )
    else
      a_learned_derivative = Float64.(Vector(learned)) .* Float64.(fact[1:length(learned)])
      Dict{String,Any}("pinn_coefficients" => a_learned_derivative)
    end

    # Write results.json — self-contained output for nn-viewer
    results = Dict(
      "representation" => String(settings.representation),
      "alpha_matrix" => vec(alpha_matrix_key),
      "benchmark_coefficients" => benchmark_series_coeffs,
      "function_error" => Float64(local_relerr),
      "objective" => Float64(loss),
      "objective_components" => Dict(
        "optimized" => ["pde", "supervised"],
        "diagnostic" => ["bc"],
        "pde" => Float64(pde),
        "bc" => Float64(bc),
        "supervised" => Float64(sup)
      ),
      "solution" => Dict(
        "x" => Float64.(collect(settings.xs)),
        "predicted" => Float64.(Vector(U[:, idx])),
        "truth" => Float64.(Vector(UT[:, idx]))
      ),
      "iteration" => iteration
    )
    merge!(results, result_payload)
    push!(all_results, results)

    if write_results_json
      results_file = joinpath(output_dir, "results.json")
      entry_id = generate_run_id(settings.optimizer)
      append_to_results_json(results_file, entry_id, results)

      @info "Results written to $results_file (run: $run_id)"
    end
    if DEBUG
      @info "PINN's representation output: $learned"
      @info "The REAL coefficients: $benchmark_series_coeffs"
    end
  end

  return loss, all_results
end

# ---------------------------------------------------------------------------
# Step 10: Export Functions
# ---------------------------------------------------------------------------

export PINNSettings, EpochBatchIterator, train_pinn, global_loss, global_loss_batched, evaluate_solution, initialize_network, io_dims

end
