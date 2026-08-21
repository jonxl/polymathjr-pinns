# CLI Reference

Runtime configuration via command-line flags.

---

## Training (`src/main.jl`)

```bash
julia --project src/main.jl [FLAGS]
```

Run with `--help` to see all options:

```bash
julia --project src/main.jl --help
```

---

## Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode` | String | `"TRAIN"` | Training mode: `TRAIN` or `GRID_SEARCH` |
| `--representation` | String | `"power_series"` | Solution representation: `power_series` or `eigenvalue` |
| `--gen-data` | Flag | `false` | Regenerate datasets via plugboard before training |
| `--data` | String | `"RANDOM"` | Dataset generation mode: `RANDOM` or `SPECIFIC` |
| `--no-snap` | Flag | snapshots ON | Disable saving weight snapshots during training |
| `--snap-every` | Int | `100` | Legacy iteration snapshot interval; epoch snapshots are controlled by `--epochs` in mini-batch mode |
| `--resume` | String | `nothing` | Path to a `.checkpoint` (raw) or `.safetensors` model file for warm-start |
| `--bins` | Int | `32` | ODEs per bin. `0` = full batch (all ODEs per iteration) |
| `--epochs` | Int | `10` | Save an intermediate checkpoint after every N complete epochs (mini-batch mode) |
| `--maxiters` | Int | `10000` | Maximum number of training iterations (gradient updates) |
| `--train-size` | Int | `1000` | Number of ODE examples to generate for the training dataset |
| `--grid-mode` | String | `"parallel"` | Grid search execution mode: `parallel` (batched across threads) or `sequential` (one config at a time) |

---

## Examples

```bash
# Default training (all defaults)
julia --project src/main.jl

# Quick test run — 500 iterations, no snapshots, full batch
julia --project src/main.jl --maxiters 500 --no-snap --bins 0

# Resume from checkpoint
julia --project src/main.jl --resume results/run-adam-02-26-26/snapshots/iter-0005000.checkpoint

# Grid search mode
julia --project src/main.jl --mode GRID_SEARCH

# Regenerate dataset then train
julia --project src/main.jl --gen-data --data RANDOM

# Threaded grid search
julia --project -t auto src/main.jl --mode GRID_SEARCH

# Train the unified eigenvalue representation
julia --project src/main.jl --representation eigenvalue
```

---

## Checkpoint Format & Conversion

Training writes checkpoints in the native raw `.checkpoint` format (Julia `Serialization`):
`results/run-{id}/model.checkpoint` plus `results/run-{id}/snapshots/iter-NNNNNNN.checkpoint`
when checkpointing is enabled.

To export the shared Hugging Face `.safetensors` format, run the explicit conversion step:

```bash
# Convert a single checkpoint
julia --project=. scripts/convert_checkpoint.jl results/run-{id}/model.checkpoint

# Convert every checkpoint in a directory
julia --project=. scripts/convert_checkpoint.jl results/run-{id}/snapshots/
```

---

## Diagnostics Dashboard (`src/explore.jl`)

### Usage

```bash
julia --project src/explore.jl [RESULTS_JSON] [LOSS_CSV] [--theme THEME]
julia --project src/explore.jl data/shared_transfer_power_series.json data/shared_transfer_eigenvalue.json
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `RESULTS_JSON` | No | Path to `training_results.json`. Auto-detected from `results/` if omitted. |
| `LOSS_CSV` | No | Path to `loss.csv`. Auto-detected from same run as JSON if omitted. |
| `PANEL_JSON` | No | One or more PanelSet JSON files emitted by `scripts/shared/` |
| `--theme NAME` | No | Colour theme: `dark` (default), `light`, `high_contrast` |
| `--help` / `-h` | No | Show help text |

### Examples

```bash
# Auto-detect latest run (dark theme)
julia --project src/explore.jl

# Specify exact paths
julia --project src/explore.jl results/run-adam-07-23-26/training_results.json results/run-adam-07-23-26/loss.csv

# Light theme
julia --project src/explore.jl --theme light

# Compare shared experiment renderings
julia --project src/explore.jl data/shared_genradius_family_power_series.json data/shared_genradius_family_eigenvalue.json
```

---

## What Stays In-File

PINN hyperparameters are stable architectural choices and remain as constants in `src/main.jl`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEURON_COUNT` | `100` | Hidden layer width |
| `SEED` | `1234` | RNG seed |
| `N` | `20` | Power series degree |
| `NUM_SUPERVISED` | `21` | Supervised coefficients |
| `NUM_POINTS` | `22` | Collocation points |
| `X_LEFT` / `X_RIGHT` | `0.0` / `1.0` | Domain bounds |
| `SUPERVISED_WEIGHT` | `1.0` | Supervised loss weight |
| `PDE_WEIGHT` | `1.0` | PDE residual loss weight |

To change these, edit `src/main.jl` directly.

---

*See also: [Quickstart](quickstart.md), [main.jl](../julia-modules/main.md)*
