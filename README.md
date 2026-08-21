# PolymathJr PINNs — power-series and eigenvalue solvers for linear ODEs

A Physics-Informed Neural Network (PINN) that learns a **representation** of the solution to an
Ordinary Differential Equation, rather than the solution function itself. Two representations are
supported, and the repository is also a study of **how the choice of representation and the
trace-determinant geometry affect what the network can learn and how well it generalizes.**

## The problem

For the family of 2nd-order linear ODEs

```
y'' - τ y' + Δ y = 0,     y(0) = 1, y'(0) = 0
```

a network maps the ODE parameters `(τ, Δ)` to a representation of the solution `u(x)`,
trained with a residual (physics) loss plus optional supervision. Initial-condition loss is logged as a diagnostic.
`(τ, Δ)` are the trace and determinant; the plane splits into **saddle / node / spiral /
center** regions depending on the roots of `r² − τr + Δ = 0`.

## The two representations

| Representation | Network output | Solution form |
|---|---|---|
| **Power series** | `[ψ₀ … ψ_N]` | `u(x) = Σ ψₙ xⁿ` (monomial basis) |
| **Eigenvalue** | `(μ, k, A, B)` | `u(x) = e^{μx}[A·C(k,x) + B·S(k,x)]` |

The network trunk is identical between them; only the input/output layer widths, the loss
triple, and the solution reconstruction differ.

## Features

- **Two solution representations**: power series and eigenvalue/exponential
- **Multi-Loss Training**: Optimizes PDE residual and supervised losses, with boundary/initial-condition diagnostics
- **GPU Acceleration**: Auto-detects CUDA GPUs, falls back to CPU transparently
- **Adam + LBFGS Optimization**: Adam active; LBFGS under investigation for convergence tuning
- **Interactive Visualization**: Native Julia + GLMakie dashboard for analyzing results

## Project Structure

```
├── src/
│   ├── main.jl              # Training entry point
│   └── explore.jl           # Diagnostics dashboard
├── architectures/
│   ├── PINN.jl              # Core PINN implementation
│   ├── PINN_RNN.jl          # RNN-based variant
│   └── PINN_specific.jl     # Specialized solver (module)
├── utils/
│   ├── plugboard.jl         # ODE dataset generation
│   ├── loss_functions.jl    # Loss computation (both representations)
│   ├── gpu_utils.jl         # GPU detection and device transfers
│   └── ...                  # Other utilities
├── viz/
│   ├── Viz.jl               # Dashboard module
│   ├── Theme.jl             # Colour themes (dark/light/high-contrast)
│   └── NNViewer.jl          # Interactive GLMakie viewer
├── scripts/                 # Standalone generalization experiments (see below)
├── data/                    # Datasets and outputs
└── docs/                    # Full documentation
```

## Experiment scripts

Standalone experiment drivers. Each is self-contained and run from the repo root; plots land
in `data/`.

```
scripts/
  PINN_specific.jl          original single-ODE power-series driver (u''' = cos πx)

  power_series/             solution as power-series coefficients:  u(x) = Σ cₙ xⁿ
    family/
      generalize_pinn.jl      family map (τ,Δ) → [c₀…c_N] over the plane
      exp_family_sum.jl       exponential (real-root) family; memorize / out-of-box / sum
    generalization/
      cross_region.jl         6×6 train-on-one-region, test-on-all transfer matrix
                              (+ per-component pde/ic/sup breakdown & bars)
      cross_region_range.jl   out-of-range × out-of-family sweeps
      extrapolate_sweep.jl    error vs. distance outside the training box
      generalize_sweep.jl     generalization sweep over the plane
    capacity/
      single_ode_memorize.jl  single-ODE memorization; loss-component & metric breakdown
      width_sweep_memorize.jl memorization vs. network width
      n_sweep.jl              error vs. series degree N
      linear_combinations.jl  fixed-sum linear combinations

  eigenvalue/               solution as eigenvalue/exponential parameters
    family/
      eig_family.jl           (τ,Δ) → exponential-form parameters for real-root regions
    generalization/
      eig_cross_region.jl      eig transfer matrix (real-root regions)
      eig_cross_region_full.jl unified form u = e^{μx}(A·C(k,x)+B·S(k,x)), all 6 regions
                              (+ component matrices, bars, power-series comparison)
      eig_showcase.jl         power-series vs. eig head-to-head across test types
      gen_radius.jl           generalization radius: disk radial profile + per-family maps
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

### Staged variant runs (one model per GPU)

```bash
# Dispatch variants across visible CUDA devices, render a per-GPU "cashier board"
julia --project=. -t auto scripts/staged_variants.jl
```

See [`docs/julia-modules/tui.md`](docs/julia-modules/tui.md) and
[`docs/julia-modules/variants.md`](docs/julia-modules/variants.md) for details.

### Generalization experiments × representations

```bash
# Run transfer, extrapolate, range, showcase, gen_radius for both reps
# concurrently via the per-GPU variant runner.
julia --project=. -t auto scripts/generalization_variants.jl
```

Subset by experiment with `--experiments transfer,extrapolate`. See
[`docs/julia-modules/generalization-variants.md`](docs/julia-modules/generalization-variants.md)
for details.

Runtime knobs (training mode, snapshots, bin size) are configurable via CLI flags — see [CLI Reference](docs/getting-started/cli-reference.md).

To run a standalone experiment instead of the main pipeline:

```bash
julia --project=. scripts/power_series/generalization/cross_region.jl
julia --project=. scripts/eigenvalue/generalization/gen_radius.jl
```

## Documentation

Full documentation is available in the [`docs/`](docs/README.md) directory:

- **[Getting Started](docs/getting-started/installation.md)** - Installation and quickstart
- **[Architecture](docs/architecture/overview.md)** - System design overview
- **[Julia Modules](docs/julia-modules/pinn.md)** - Code documentation
- **[Tutorials](docs/tutorials/hyperparameter-search.md)** - Step-by-step guides
- **[API Reference](docs/api-reference/julia-api.md)** - Function signatures

## How It Works

1. **Dataset Generation**: `plugboard.jl` creates ODEs and computes analytical power series coefficients
2. **Network Training**: PINN learns to predict a representation that satisfies the ODE constraints
3. **Loss Function**: Optimizes PDE residual and supervised loss, while logging BC/IC diagnostics
4. **Evaluation**: Compares predicted vs true solutions on benchmark ODEs

## Diagnostics Dashboard

```bash
# Launch the interactive GLMakie dashboard
julia --project src/explore.jl

# With a specific run
julia --project src/explore.jl results/run-adam-06-03-26/training_results.json results/run-adam-06-03-26/loss.csv

# Choose a colour theme
julia --project src/explore.jl --theme light
```

The training dashboard shows 8 diagnostic plots (solution comparison, coefficient analysis, loss curves) with an interactive iteration range slider. Shared experiment JSON files from `scripts/shared/` open in the generic panel viewer.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Julia 1.9+ |
| Neural Networks | Lux.jl |
| Optimization | Optimization.jl (Adam active, LBFGS planned) |
| Autodiff | Zygote.jl |
| GPU | CUDA.jl (auto-detected) |
| Visualization | Julia / GLMakie.jl (pipeline), Plots.jl (scripts) |

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
