# Project Structure

## Directory Layout

```
polymathjr-pinns-jon-jeet/
├── src/
│   └── main.jl                 # Entry point
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
├── scripts/
│   ├── main.py                 # Python examples
│   └── visualizer.py           # Interactive visualization
├── data/
│   ├── training_dataset.json
│   ├── benchmark_dataset.json
│   └── training-run-*/         # Output directories
├── docs/                       # Documentation
├── Project.toml                # Julia dependencies
└── README.md
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/main.jl` | Orchestrates training runs |
| `architectures/PINN.jl` | Core PINN with `PINNSettings`, `train_pinn()`, `global_loss()` |
| `utils/plugboard.jl` | Generates ODEs and computes power series coefficients |
| `utils/loss_functions.jl` | PDE, BC, supervised loss functions |
| `utils/gpu_utils.jl` | GPU detection, device transfer utilities |
| `scripts/visualizer.py` | `GeneralizedVisualizer`, `PowerSeriesVisualizer` classes |

---

*See also: [Architecture Overview](../architecture/overview.md)*
