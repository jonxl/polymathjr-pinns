# Quickstart

Run your first PINN training.

## Run Training

```bash
julia --project src/main.jl
```

This will:
1. Load ODE training dataset
2. Initialize neural network
3. Train with Adam optimizer
4. Save checkpoint weights as `.safetensors` files when checkpointing is enabled
5. Save the final trained MLP weights to `model.safetensors`
6. Evaluate and save results

Pass flags to customize the run without editing source:

```bash
# Quick test — no snapshots, full batch
julia --project src/main.jl --no-snap --bins 0

# Resume from a checkpoint
julia --project src/main.jl --resume results/run-adam-02-26-26/snapshots/iter-0005000.safetensors
```

See the [CLI Reference](cli-reference.md) for all available flags.

---

## Output Structure

Current training runs write under `results/run-{id}/`:

```
results/run-adam-MM-DD-YY/
├── model.safetensors          # final trained MLP weights
├── training_results.json      # metadata, final objective, coefficients, checkpoints
├── loss.csv                   # sampled loss history
└── snapshots/
    └── iter-NNNNNNN.safetensors  # optional checkpoint weights
```

---

## View Results

### Interactive Dashboard

```bash
# Auto-detect latest run
julia --project src/explore.jl

# Specify run paths
julia --project src/explore.jl results/run-adam-06-03-26/training_results.json results/run-adam-06-03-26/loss.csv

# Light theme
julia --project src/explore.jl --theme light

# Show help
julia --project src/explore.jl --help
```

The dashboard opens an interactive GLMakie window with:

- **Solution Comparison** — benchmark vs PINN power series
- **Absolute Error** — pointwise solution error (log scale)
- **Coefficient Comparison** — benchmark vs PINN coefficients
- **Coefficient Error** — per-coefficient absolute error (log scale)
- **4 Loss Curves** — total, BC, PDE, supervised loss over training
- **Iteration Range Slider** — single bar with two handles (start, end)

Closing the window automatically exits the Julia process.

---

## Configuration

**Runtime knobs** (training mode, snapshots, bin size) are set via CLI flags — see [CLI Reference](cli-reference.md).

**PINN hyperparameters** are in-file constants in `src/main.jl`:

```julia
NEURON_COUNT = 100            # Hidden layer width
MAXITERS = 10000              # Training iterations
N = 10                        # Power series degree
SUPERVISED_WEIGHT = 1.0f0     # Supervised loss weight
PDE_WEIGHT = 1.0f0            # PDE residual weight
```

---

*Next: [Project Structure](project-structure.md)*
