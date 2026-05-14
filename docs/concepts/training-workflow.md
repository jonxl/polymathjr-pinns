# Training Workflow

Complete pipeline from ODE definition to trained model.

---

## Pipeline Overview

```
1. Dataset Generation
   └─→ plugboard.jl generates training_dataset.json

2. Initialization
   └─→ Load datasets, create PINNSettings

3. Device Detection
   └─→ Auto-detect GPU (CUDA) or fall back to CPU

4. Network Setup
   └─→ initialize_network() creates Lux model, transfers to GPU if available

5. Adam Training (LBFGS under investigation)
   └─→ Configurable iterations, optimization

6. Evaluation
   └─→ Parameters transferred back to CPU, benchmark testing, plot generation

7. Output
   └─→ CSV loss history, PNG plots
```

---

## Detailed Steps

### Step 1: Dataset Generation

```julia
# In plugboard.jl
settings = Settings(
    ode_order = 2,
    poly_degree = 0,
    dataset_size = 10,
    num_of_terms = 15
)
generate_random_ode_dataset(settings, batch_index)
```

**Output:** `training_dataset.json` with ODEs and true coefficients

---

### Step 2: Create PINNSettings

```julia
settings = PINNSettings(
    neuron_num = 50,
    seed = 42,
    ode_matrices = load_training_data(),
    maxiters_adam = 10000,
    n_terms_for_power_series = 10,
    # ... other parameters
)
```

---

### Step 3: Device Detection & Network Setup

```julia
# Automatic GPU detection in train_pinn()
use_gpu = GPUUtils.is_gpu_available()  # true if CUDA functional

coeff_net, p_net, st = initialize_network(settings; use_gpu=use_gpu)
```

**Architecture:**
- Input: flattened ODE matrix + initial conditions
- Hidden: 4 layers × `neuron_num` neurons (σ activation)
- Output: N+1 coefficients
- Parameters transferred to GPU via `CUDA.cu()` when available

---

### Step 4: Adam Training

```julia
# In train_pinn()
opt_func = OptimizationFunction(loss_wrapper, AutoZygote())
prob = OptimizationProblem(opt_func, p_net)
result = solve(prob, Adam(0.001f0), maxiters=settings.maxiters_lbfgs)
```

**Purpose:** Optimization with configurable iteration count. Loss functions use vectorized matrix operations for GPU compatibility (no scalar indexing).

> **Note:** LBFGS code is present in `train_pinn` but currently commented out. Further work is needed to understand its convergence behavior before re-enabling.

---

### Step 6: Evaluation

```julia
evaluate_solution(
    settings,
    p_trained,
    coeff_net,
    st,
    benchmark_dataset,
    output_directory
)
```

**Generates:**
- Solution comparison plot
- Coefficient comparison plot
- Error analysis plot

---

### Step 7: Output Files

```
data/training-run-N/batch-01/
├── iteration_output.csv
├── solution_plot.png
├── coefficient_plot.png
└── error_plot.png
```

---

## Loss During Training

Each iteration logs to `loss.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `iteration` | Training iteration number |
| `total` | Weighted sum of all losses |
| `pde` | ODE residual loss |
| `bc` | Boundary condition loss |
| `supervised` | Coefficient MSE |

---

## Checkpointing (Snapshots)

Training saves the final trained MLP weights to `results/run-{id}/model.safetensors`.

When checkpointing is enabled, intermediate weights are saved to `results/run-{id}/snapshots/iter-NNNNNNN.safetensors`. These checkpoints enable:

- **Warm-start**: Resume training from a saved checkpoint with `--resume <path>`
- **Replay**: Evaluate the model at each saved iteration without retraining

Snapshot behavior is controlled via CLI flags:

| Flag | Effect |
|------|--------|
| `--no-snap` | Disable snapshot saving entirely |
| `--snap-every N` | Legacy iteration interval; mini-batch snapshots use `--epochs` |
| `--epochs N` | Save after every N epochs in mini-batch mode (default: 10) |
| `--resume <path>` | Warm-start from a `.safetensors` snapshot file; legacy `.bin` files are still readable |

`--no-snap` disables intermediate checkpoints only. The final `model.safetensors` is still written.

See the [CLI Reference](../getting-started/cli-reference.md) for all flags.

---

*See also: [PINN.jl](../julia-modules/pinn.md)*
