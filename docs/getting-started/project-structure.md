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
│   ├── variants.jl             # Staged training variant runner
│   ├── generalization_variants.jl
│   ├── generalization_geometry.jl
│   ├── geometry_experiments.jl
│   ├── geometry_comparison.jl
│   ├── safetensors_utils.jl    # Portable checkpoint export/load helpers
│   ├── ProgressBar.jl          # Progress tracking
│   ├── binary_search_on_weights.jl
│   └── two_d_grid_search_*.jl  # Hyperparameter search
├── viz/
│   ├── Viz.jl                  # Dashboard module (top-level)
│   ├── Theme.jl                # Colour themes (dark/light/high-contrast)
│   ├── NNViewer.jl             # Interactive GLMakie PINN viewer
│   ├── PanelSpec.jl            # JSON schema/contract for experiment panels
│   ├── Panels.jl               # GLMakie panel renderers
│   └── PanelViewer.jl          # Multi-PanelSet viewer
├── scripts/
│   ├── staged_variants.jl      # One-model-per-GPU staged training runs
│   ├── generalization_variants.jl
│   ├── convert_checkpoint.jl
│   ├── shared/
│   ├── power_series/
│   └── eigenvalue/
├── data/
│   ├── training_dataset.json
│   ├── benchmark_dataset.json
│   └── training-run-*/         # Output directories
├── results/
│   └── run-adam-*/             # Training run outputs
├── test/                       # Regression tests
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
| `utils/training_schemes.jl` | Main training loop, batching, snapshots, and checkpoint resume |
| `utils/variants.jl` | Staged variant jobs and per-device dispatch |
| `utils/generalization_variants.jl` | Cross-product runner for representations and generalization experiments |
| `utils/generalization_geometry.jl` | Geometry experiment definitions and controls |
| `utils/geometry_experiments.jl` | Shared geometry experiment execution |
| `utils/geometry_comparison.jl` | Panel comparisons for geometry outputs |
| `viz/Viz.jl` | Top-level dashboard module (`Viz.explore()`) |
| `viz/NNViewer.jl` | Interactive GLMakie PINN results viewer |
| `viz/PanelSpec.jl` | Serializable Panel/PanelSet contract for experiment output |
| `viz/Panels.jl` | GLMakie renderers for heatmaps, bars, lines, scatters, and grids |
| `viz/PanelViewer.jl` | Viewer for one or more PanelSet JSON files |
| `viz/Theme.jl` | Colour theme system (dark, light, high-contrast) |

---

*See also: [Architecture Overview](../architecture/overview.md)*
