# PINN.jl

Core PINN implementation for learning either power-series coefficients or unified eigenvalue parameters.

**Location:** `architectures/PINN.jl`

---

## PINNSettings Struct

```julia
struct PINNSettings
    neuron_num::Int              # Neurons per hidden layer
    seed::Int                    # Random seed
    ode_matrices::Dict{Any,Any}  # ODE coefficient matrices
    maxiters_lbfgs::Int          # Adam iterations
    n_terms_for_power_series::Int # Degree N of power series
    num_supervised::Int          # Coefficients for supervision
    num_points::Int              # Collocation points
    x_left::Float32              # Domain left boundary
    x_right::Float32             # Domain right boundary
    supervised_weight::Float32   # Weight for supervised loss
    pde_weight::Float32          # Weight for PDE loss
    xs::Any                      # Collocation point locations
    optimizer::String
    representation::Symbol       # :power_series or :eigenvalue
end
```

---

## Key Functions

### `initialize_network(settings; use_gpu=false)`

Creates neural network architecture and optionally transfers parameters to GPU.

```julia
initialize_network(settings::PINNSettings; use_gpu::Bool=false) → (network, params, state)
```

**Architecture:**
- 4 hidden layers with configurable neuron count
- Sigmoid activation function
- Input/output layers are sized by `io_dims(settings)` for the selected representation

When `use_gpu=true`, parameters are transferred to GPU via `CUDA.cu()`.

---

### `loss_fn(p_net, data, coeff_net, st, ode_matrix_flat, boundary_condition, settings, use_gpu=false)`

Computes loss for a single ODE. Transfers all inputs to the correct device (GPU/CPU) before the forward pass.

```julia
loss_fn(...) → (total_loss, loss_bc, loss_pde, loss_supervised)
```

**Components:**
- PDE residual loss (vectorized matrix multiply, GPU-compatible)
- Boundary/initial-condition diagnostic loss
- Supervised loss: coefficient-space for power series, solution-space for eigenvalue

The optimized objective is:

```julia
pde_weight * num_supervised * pde + supervised_weight * supervised
```

BC/IC loss is returned and logged, but it is not part of the optimized objective.

---

### `global_loss(p_net, settings, coeff_net, st)`

Aggregates loss across all training examples.

```julia
    global_loss(...) → (mean_loss, aggregated_components)
```

---

### `train_pinn(settings, output_dir; on_milestone=nothing, on_interrupt=nothing, batch_size=0, snapshot_epoch_interval=10)`

Auto-detects GPU and trains with Adam. LBFGS is available but currently disabled pending convergence investigation. Returns CPU parameters regardless of training device. Writes `loss.csv` to `output_dir` after training.

```julia
train_pinn(settings::PINNSettings, output_dir; on_milestone=nothing, on_interrupt=nothing, batch_size=0, snapshot_epoch_interval=10) → (trained_params, network, state, run_id, history)
```

**Behavior:**
- Checks `GPUUtils.is_gpu_available()` at start
- Transfers network parameters to GPU if available
- Runs Adam optimization for `maxiters_lbfgs` iterations
- LBFGS code is present but commented out (needs further work on convergence)
- Writes loss history to `<output_dir>/loss.csv` (rows=iterations, columns: `iteration`, `total`, `bc`, `pde`, `supervised`)
- Transfers trained parameters back to CPU before returning
- Optionally calls `on_milestone(params, iteration, net, state, run_id)` at configured epoch boundaries for checkpointing
- On `InterruptException`, calls `on_interrupt(params, iteration, net, state, run_id)` (if set) so the caller can save an interrupted-state checkpoint

---

### `evaluate_solution(settings, p_trained, coeff_net, st, benchmark_dataset, output_dir, run_id)`

Evaluates a trained model and writes representation-aware JSON payloads.

```julia
evaluate_solution(...) → (objective, results)
```

**Outputs:**
- Solution-space relative error
- Power-series derivative coefficients or eigenvalue parameters
- Optimized and diagnostic loss components

---

## Example Usage

```julia
settings = PINNSettings(
    neuron_num = 50,
    seed = 42,
    ode_matrices = training_data,
    maxiters_lbfgs = 10000,
    n_terms_for_power_series = 10,
    num_supervised = 5,
    num_points = 20,
    x_left = 0.0f0,
    x_right = 1.0f0,
    supervised_weight = 0.1f0,
    pde_weight = 1.0f0,
    xs = collect(range(0, 1, 20)),
    optimizer = "adam",
    representation = :power_series
)

p_trained, net, st = train_pinn(settings, "output.csv")
```

---

*See also: [Training Workflow](../concepts/training-workflow.md)*
