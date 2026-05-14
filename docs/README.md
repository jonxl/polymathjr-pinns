# PolyMathJr PINN Documentation

Documentation for the Physics-Informed Neural Network (PINN) Power Series ODE Solver.

## Overview

This project implements a PINN that learns power series coefficients to approximate solutions of Ordinary Differential Equations (ODEs). The neural network outputs coefficients that form a power series satisfying the ODE.

---

## Quick Navigation

### Getting Started
- [Installation](getting-started/installation.md) - Julia & Python setup
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

### Python Modules
- [visualizer.py](python-modules/visualizer.md) - Interactive visualization
- [main.py](python-modules/main-examples.md) - Examples

### Concepts
- [Hyperparameter Tuning](concepts/hyperparameter-tuning.md) - Tuning strategies
- [Training Workflow](concepts/training-workflow.md) - Complete pipeline

### Data Formats
- [Training Dataset](data-formats/training-dataset.md) - JSON format
- [Benchmark Dataset](data-formats/benchmark-dataset.md) - Evaluation data
- [Output CSV](data-formats/output-csv.md) - Loss history
- [Coefficients JSON](data-formats/coefficients-json.md) - Visualization format

### API Reference
- [Julia API](api-reference/julia-api.md) - Function signatures
- [Python API](api-reference/python-api.md) - Class/function signatures

### Tutorials
- [Hyperparameter Search](tutorials/hyperparameter-search.md) - Grid/binary search
- [Visualization Guide](tutorials/visualization-guide.md) - Python dashboard
- [Scaling Experiments](tutorials/scaling-experiments.md) - Neuron/iteration studies

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | Julia 1.9+ |
| Neural Networks | Lux.jl |
| Optimization | Optimization.jl (Adam, LBFGS) |
| Autodiff | Zygote.jl |
| Visualization | Python 3.14+ / Matplotlib |

---

*Last updated: January 2026*
