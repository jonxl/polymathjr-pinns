# PINN Power Series ODE Solver

A Physics-Informed Neural Network (PINN) that learns power series coefficients to approximate solutions of Ordinary Differential Equations (ODEs).

## Overview

Instead of learning the solution function directly, this PINN outputs the coefficients of a truncated power series. The neural network learns to predict coefficients such that the resulting power series satisfies the ODE.

## Features

- **Power Series Learning**: Neural network outputs power series coefficients
- **Multi-Loss Training**: Combines PDE residual, boundary conditions, and supervised losses
- **GPU Acceleration**: Auto-detects CUDA GPUs, falls back to CPU transparently
- **Adam + LBFGS Optimization**: Adam active; LBFGS under investigation for convergence tuning
- **Interactive Visualization**: Native Julia + GLMakie dashboard for analyzing results

## Project Structure

```
├── src/
│   ├── main.jl              # Training entry point
│   └── explore.jl            # Diagnostics dashboard
├── architectures/
│   ├── PINN.jl              # Core PINN implementation
│   ├── PINN_RNN.jl          # RNN-based variant
│   └── PINN_specific.jl     # Specialized solver
├── utils/
│   ├── plugboard.jl         # ODE dataset generation
│   ├── loss_functions.jl    # Loss computation
│   ├── gpu_utils.jl         # GPU detection and device transfers
│   └── ...                  # Other utilities
├── viz/
│   ├── Viz.jl               # Dashboard module
│   ├── Theme.jl             # Colour themes (dark/light/high-contrast)
│   └── NNViewer.jl          # Interactive GLMakie viewer
├── data/                    # Datasets and outputs
└── docs/                    # Full documentation
```

## Quick Start

```bash
# 1. Clone the repository
git clone git@github.com:jonxlegasa/polymathjr-pinns-jon-jeet.git
cd polymathjr-pinns-jon-jeet

# 2. Install Julia dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 3. Run training
julia --project src/main.jl

# 4. See all runtime options
julia --project src/main.jl --help
```

Runtime knobs (training mode, snapshots, bin size) are configurable via CLI flags — see [CLI Reference](docs/getting-started/cli-reference.md).

## Documentation

Full documentation is available in the [`docs/`](docs/README.md) directory:

- **[Getting Started](docs/getting-started/installation.md)** - Installation and quickstart
- **[Architecture](docs/architecture/overview.md)** - System design overview
- **[Julia Modules](docs/julia-modules/pinn.md)** - Code documentation
- **[Tutorials](docs/tutorials/hyperparameter-search.md)** - Step-by-step guides
- **[API Reference](docs/api-reference/julia-api.md)** - Function signatures

## How It Works

1. **Dataset Generation**: `plugboard.jl` creates ODEs and computes analytical power series coefficients
2. **Network Training**: PINN learns to predict coefficients that satisfy ODE constraints
3. **Loss Function**: Combines PDE residual, boundary conditions, and supervised coefficient loss
4. **Evaluation**: Compares predicted vs true coefficients on benchmark ODE

## Diagnostics Dashboard

```bash
# Launch the interactive GLMakie dashboard
julia --project src/explore.jl

# With a specific run
julia --project src/explore.jl results/run-adam-06-03-26/training_results.json results/run-adam-06-03-26/loss.csv

# Choose a colour theme
julia --project src/explore.jl --theme light
```

The dashboard shows 8 diagnostic plots (solution comparison, coefficient analysis, loss curves) with an interactive iteration range slider.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Julia 1.9+ |
| Neural Networks | Lux.jl |
| Optimization | Optimization.jl (Adam active, LBFGS planned) |
| Autodiff | Zygote.jl |
| GPU | CUDA.jl (auto-detected) |
| Visualization | Julia / GLMakie.jl |

## License

MIT

## Citation

If you use this code in your research, please cite:

```bibtex
@software{polymathjr_pinn,
  title = {PINN Power Series ODE Solver},
  author = {PolyMathJr Team},
  year = {2026},
  url = {https://github.com/jonxlegasa/polymathjr-pinns-jon-jeet}
}
```
