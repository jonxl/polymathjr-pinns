# Project Structure

## Directory Layout

```
polymathjr-pinns-jon-jeet/
├── src/
│   ├── main.jl                 # Training entry point
│   └── explore.jl              # Diagnostics dashboard entry point
├── architectures/
│   ├── PINN.jl                 # Core feedforward PINN
│   ├── PINN_RNN.jl             # RNN-based variant
│   └── PINN_specific.jl        # Specialized 1st-order solver
├── utils/
│   ├── plugboard.jl            # ODE dataset generation
│   ├── helper_funcs.jl         # Utilities
│   ├── loss_functions.jl       # Loss computation
│   ├── gpu_utils.jl            # GPU detection and device transfers
│   ├── training_schemes.jl     # Training strategies
│   ├── ProgressBar.jl          # Progress tracking
│   ├── binary_search_on_weights.jl
│   └── two_d_grid_search_*.jl  # Hyperparameter search
├── viz/
│   ├── Viz.jl                  # Dashboard module (top-level)
│   ├── Theme.jl                # Colour themes (dark/light/high-contrast)
│   └── NNViewer.jl             # Interactive GLMakie PINN viewer
├── data/
│   ├── training_dataset.json
│   ├── benchmark_dataset.json
│   └── training-run-*/         # Output directories
├── results/
│   └── run-adam-*/             # Training run outputs
├── docs/                       # Documentation
├── Project.toml                # Julia dependencies
└── README.md
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/main.jl` | Orchestrates training runs |
| `src/explore.jl` | Launches interactive diagnostics dashboard |
| `architectures/PINN.jl` | Core PINN with `PINNSettings`, `train_pinn()`, `global_loss()` |
| `utils/plugboard.jl` | Generates ODEs and computes power series coefficients |
| `utils/loss_functions.jl` | PDE, BC, supervised loss functions |
| `utils/gpu_utils.jl` | GPU detection, device transfer utilities |
| `viz/Viz.jl` | Top-level dashboard module (`Viz.explore()`) |
| `viz/NNViewer.jl` | Interactive GLMakie PINN results viewer |
| `viz/Theme.jl` | Colour theme system (dark, light, high-contrast) |

---

*See also: [Architecture Overview](../architecture/overview.md)*
