# CLI Reference

Runtime configuration for `src/main.jl` via command-line flags.

---

## Usage

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
| `--gen-data` | Flag | `false` | Regenerate datasets via plugboard before training |
| `--data` | String | `"RANDOM"` | Dataset generation mode: `RANDOM` or `SPECIFIC` |
| `--no-snap` | Flag | snapshots ON | Disable saving weight snapshots during training |
| `--snap-every` | Int | `100` | Legacy iteration snapshot interval; epoch snapshots are controlled by `--epochs` in mini-batch mode |
| `--resume` | String | `nothing` | Path to `.safetensors` snapshot file for warm-start; legacy `.bin` files are still readable |
| `--bins` | Int | `32` | ODEs per bin. `0` = full batch (all ODEs per iteration) |
| `--epochs` | Int | `10` | Save an intermediate checkpoint after every N complete epochs (mini-batch mode) |
| `--maxiters` | Int | `10000` | Maximum number of training iterations (gradient updates) |

---

## Examples

```bash
# Default training (all defaults)
julia --project src/main.jl

# Quick test run — 500 iterations, no snapshots, full batch
julia --project src/main.jl --maxiters 500 --no-snap --bins 0

# Resume from checkpoint
julia --project src/main.jl --resume results/run-adam-02-26-26/snapshots/iter-0005000.safetensors

# Grid search mode
julia --project src/main.jl --mode GRID_SEARCH

# Regenerate dataset then train
julia --project src/main.jl --gen-data --data RANDOM

# Threaded grid search
julia --project -t auto src/main.jl --mode GRID_SEARCH
```

---

## What Stays In-File

PINN hyperparameters are stable architectural choices and remain as constants in `src/main.jl`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEURON_COUNT` | `100` | Hidden layer width |
| `SEED` | `1234` | RNG seed |
| `N` | `10` | Power series degree |
| `NUM_SUPERVISED` | `10` | Supervised coefficients |
| `NUM_POINTS` | `10` | Collocation points |
| `X_LEFT` / `X_RIGHT` | `0.0` / `1.0` | Domain bounds |
| `SUPERVISED_WEIGHT` | `1.0` | Supervised loss weight |
| `BC_WEIGHT` | `1.0` | Boundary condition loss weight |
| `PDE_WEIGHT` | `1.0` | PDE residual loss weight |

To change these, edit `src/main.jl` directly.

---

*See also: [Quickstart](quickstart.md), [main.jl](../julia-modules/main.md)*
