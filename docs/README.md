# PolyMathJr PINN Documentation

Documentation for the Physics-Informed Neural Network (PINN) ODE solver and generalization study.

## Overview

This project trains PINNs to learn representations of linear ODE solutions. The main training path supports both a power-series head and a unified eigenvalue/exponential head for the trace-determinant family, with diagnostics for how representation and geometry affect generalization.

---

## Quick Navigation

### Getting Started
- [Installation](getting-started/installation.md) - Julia setup
- [Quickstart](getting-started/quickstart.md) - First training run
- [CLI Reference](getting-started/cli-reference.md) - Command-line flags
- [Project Structure](getting-started/project-structure.md) - Codebase layout

### Architecture
- [Overview](architecture/overview.md) - System architecture
- [Data Flow](architecture/data-flow.md) - Data pipeline

### Julia Modules
- [main.jl](julia-modules/main.md) - Entry point
- [PINN.jl](julia-modules/pinn.md) - Core implementation
- [PINN_RNN.jl](julia-modules/pinn-rnn.md) - RNN variant
- [PINN_specific.jl](julia-modules/pinn-specific.md) - Specialized solver
- [plugboard.jl](julia-modules/plugboard.md) - Dataset generation
- [helper_funcs.jl](julia-modules/helper-funcs.md) - Utilities
- [training_schemes.jl](julia-modules/training-schemes.md) - Training strategies
- [variants.jl](julia-modules/variants.md) - Staged GPU variant runner
- [generalization_variants.jl](julia-modules/generalization-variants.md) - Representation × experiment runner
- [tui.jl](julia-modules/tui.md) - Terminal progress boards
- [ProgressBar.jl](julia-modules/progress-bar.md) - Progress rendering utilities

### Viz Modules
- [Viz.jl](viz-modules/viz.md) - Dashboard module
- [Theme.jl](viz-modules/theme.md) - Colour themes
- [NNViewer.jl](viz-modules/nnviewer.md) - Interactive GLMakie training-run viewer

### Concepts
- [Hyperparameter Tuning](concepts/hyperparameter-tuning.md) - Tuning strategies
- [Training Workflow](concepts/training-workflow.md) - Complete pipeline
- [Power-Series Loss Audit](concepts/power-series-loss-audit.md) - Coefficient conventions, factorials, residual averaging, and saved loss evidence

### Data Formats
- [Training Dataset](data-formats/training-dataset.md) - JSON format
- [Benchmark Dataset](data-formats/benchmark-dataset.md) - Evaluation data
- [Output CSV](data-formats/output-csv.md) - Loss history
- [Coefficients JSON](data-formats/coefficients-json.md) - Visualization format

### Tutorials
- [Hyperparameter Search](tutorials/hyperparameter-search.md) - Grid/binary search
- [Visualization Guide](tutorials/visualization-guide.md) - GLMakie dashboard
- [Scaling Experiments](tutorials/scaling-experiments.md) - Neuron/iteration studies

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Julia 1.9+ |
| Neural Networks | Lux.jl |
| Optimization | Optimization.jl / OptimizationOptimisers.jl, OptimizationOptimJL.jl |
| Autodiff | Zygote.jl |
| GPU | CUDA.jl |
| Visualization | GLMakie.jl, Plots.jl |

---

*Last updated: September 2026*
