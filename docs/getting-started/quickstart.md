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

```
data/training-run-1/
├── batch-01/
│   ├── iteration_output.csv    # Loss history
│   ├── solution_plot.png       # Solution comparison
│   ├── coefficient_plot.png    # Coefficient comparison
│   └── error_plot.png          # Error analysis
└── metadata.json
```

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

### Loss CSV Format

```csv
loss_type,iter_1,iter_2,...
total_loss,0.95,0.82,...
total_loss_bc,0.30,0.25,...
total_loss_pde,0.50,0.42,...
total_loss_supervised,0.15,0.15,...
```

### Interactive Visualization

```bash
cd scripts
source .venv/bin/activate
python main.py
```

---

## Configuration

**Runtime knobs** (training mode, snapshots, bin size) are set via CLI flags — see [CLI Reference](cli-reference.md).

**PINN hyperparameters** are in-file constants in `src/main.jl`:

```julia
NEURON_COUNT = 100            # Hidden layer width
MAXITERS = 10000              # Training iterations
N = 10                        # Power series degree
SUPERVISED_WEIGHT = 1.0f0     # Supervised loss weight
BC_WEIGHT = 1.0f0             # Boundary condition weight
PDE_WEIGHT = 1.0f0            # PDE residual weight
```

---

*Next: [Project Structure](project-structure.md)*
