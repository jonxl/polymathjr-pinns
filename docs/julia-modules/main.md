# main.jl

Entry point for training runs.

**Location:** `src/main.jl`

---

## Functions

### `parse_commandline()`

Parses CLI flags and returns a `Dict{String,Any}` of runtime settings. Called at module load time — the returned values populate global config variables (`TRAINING_MODE`, `SAVE_SNAPSHOTS`, `BIN_SIZE`, etc.).

See [CLI Reference](../getting-started/cli-reference.md) for the full flag list.

---

### `create_training_run_dirs(run_number, batch_size)`

Creates output directory structure for a training run.

```julia
create_training_run_dirs(run_number::Int64, batch_size::Any)
```

**Creates:**
```
data/training-run-{run_number}/
├── batch-01/
├── batch-02/
└── ...
```

---

### `init_batches(batch_sizes)`

Initializes and generates ODE datasets for each batch size.

```julia
init_batches(batch_sizes::Array{Int})
```

Uses `plugboard.jl` to generate random ODEs with analytical solutions.

---

### `run_training_sequence(batch_sizes)`

Main orchestration function.

```julia
run_training_sequence(batch_sizes::Array{Int})
```

**Steps:**
1. Initialize batches (dataset generation)
2. Create `PINNSettings` for each batch
3. Call `train_pinn()` for training
4. Call `evaluate_solution()` for benchmarking

---

## Configuration

Runtime settings (training mode, snapshots, bin size, etc.) are controlled via CLI flags — see the [CLI Reference](../getting-started/cli-reference.md) for the full list.

PINN hyperparameters (`NEURON_COUNT`, `SEED`, `MAXITERS`, loss weights, etc.) remain as in-file constants.

```julia
batch = [1000]  # Number of ODEs to generate / train on
```

---

*See also: [CLI Reference](../getting-started/cli-reference.md), [PINN.jl](pinn.md), [plugboard.jl](plugboard.md)*
