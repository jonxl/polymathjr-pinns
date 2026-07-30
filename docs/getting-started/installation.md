# Installation

Set up Julia environment for training and diagnostics.

## Prerequisites

- Julia 1.9+
- Git

---

## Julia Setup

### 1. Clone Repository

```bash
git clone git@github.com:jonxlegasa/polymathjr-pinns-jon-jeet.git
cd polymathjr-pinns-jon-jeet
```

### 2. Install Dependencies

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Key packages installed:

| Package | Purpose |
|---------|---------|
| ArgParse | CLI argument parsing |
| Lux | Neural network layers |
| Optimization | Optimization framework |
| OptimizationOptimJL | LBFGS optimizer (under investigation) |
| OptimizationOptimisers | Adam optimizer |
| Zygote | Automatic differentiation |
| JSON, CSV | Data I/O |
| GLMakie | Interactive diagnostics dashboard |
| Plots | Static visualization |
| TaylorSeries | Power series operations |
| CUDA | GPU acceleration (auto-detected) |

### 3. Verify

```bash
julia --project=. -e 'using Lux, Optimization; println("Success!")'
```

---

## GPU Setup

CUDA.jl is included in `Project.toml` and installed automatically by `Pkg.instantiate()`. To verify your GPU is detected:

```bash
julia --project=. -e 'using CUDA; println(CUDA.functional() ? "GPU: $(CUDA.name(CUDA.device()))" : "No GPU detected — will train on CPU")'
```

Training auto-detects GPU availability and falls back to CPU if no GPU is present.

---

*Next: [Quickstart](quickstart.md)*
