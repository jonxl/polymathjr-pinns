# Tutorial: Visualization Guide

Using the native Julia + GLMakie dashboard to analyze PINN training results.

---

## Overview

This tutorial covers:
1. Launching the interactive dashboard
2. Understanding the 8 diagnostic plots
3. Using the iteration range slider
4. Choosing colour themes
5. Loading specific training runs

---

## Prerequisites

The dashboard is built into the project — **no extra dependencies needed**.  GLMakie is already listed in `Project.toml`.

```bash
# Make sure dependencies are installed
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

---

## Launching the Dashboard

### Auto-Detect Latest Run

```bash
julia --project src/explore.jl
```

This scans `results/` for the most recent `run-*/` directory containing a `training_results.json` and opens it.

### Specify a Run

```bash
julia --project src/explore.jl results/run-adam-06-03-26/training_results.json results/run-adam-06-03-26/loss.csv
```

### Choose a Theme

```bash
julia --project src/explore.jl --theme light
julia --project src/explore.jl --theme high_contrast
```

Available themes: `dark` (default), `light`, `high_contrast`.

### From the REPL

```julia
include("viz/Viz.jl")
using .Viz
Viz.explore("results/run-adam-06-03-26/training_results.json",
            "results/run-adam-06-03-26/loss.csv";
            theme = Viz.LIGHT_THEME)
```

---

## Dashboard Layout

The window contains 8 diagnostic panels in a 4×2 grid:

| Row | Left Panel | Right Panel |
|-----|-----------|-------------|
| 1 | **ODE Solution Comparison** — benchmark (dashed cyan) vs PINN (solid orange) power series | **Absolute Error of Solution** — `|benchmark - PINN|` on log y-axis |
| 2 | **Coefficient Comparison** — benchmark (cyan) vs PINN (orange) coefficients | **Coefficient Error** — `|benchmark - PINN|` per coefficient (log scale) |
| 3 | **Total Loss** — full training history with window highlight | **BC Loss** — boundary condition loss |
| 4 | **PDE Loss** — physics residual loss | **Supervised Loss** — supervised coefficient loss |

### Controls

- **Iteration Range Slider** (bottom) — single bar with two drag handles:
  - **Left handle** (dashed yellow line): iteration window start
  - **Right handle** (solid yellow line): snapshot selector — non-loss plots show the PINN state at the nearest milestone ≤ this iteration
  - The yellow span between handles highlights the active window on loss plots
- **Status Bar** — shows the current snapshot iteration and objective value
- **Title Bar** — shows the current right-handle iteration position

---

## Understanding the Plots

### 1. Solution Comparison

Shows `u(x) = Σ c_n x^n / n!` evaluated over the domain.

- **Cyan dashed:** Benchmark (true) power series
- **Orange solid:** PINN prediction at the selected snapshot
- **Good fit:** Lines overlap closely

### 2. Absolute Error of Solution

`|Benchmark(x) - PINN(x)|` on log scale.

- **Good:** Error stays below 0.01 across the domain
- **Warning:** Error growth near domain boundaries indicates poor extrapolation

### 3. Coefficient Comparison

Direct comparison of power series coefficients `c_n`.

- **Good:** Early coefficients match closely; later coefficients may diverge
- **Red flag:** First coefficient (IC) doesn't match → training didn't converge

### 4. Coefficient Error

Per-coefficient `|Benchmark - PINN|` on log scale.

- **Good:** Error < 0.01 for early coefficients
- **Expected:** Error grows for higher indices (higher-order terms are harder)

### 5–8. Loss Curves

Full training loss history on log scale.

- **Solid coloured line:** Loss value over iterations
- **Yellow span:** Active window between start and end handles
- **Dashed yellow vline:** Start handle position
- **Solid yellow vline:** End handle (snapshot) position
- **Good:** Monotonic decrease; final loss < 1.0

---

## Interactive Workflow

### Scrubbing Through Training

1. Drag the **right handle** to scrub through training snapshots
2. Solution and coefficient plots update in real time
3. Loss curves always show the full history; the window highlight follows the handles

### Zooming Into Loss Curves

1. Drag the **left handle** to narrow the iteration window
2. Use this to focus on early training (rapid loss drop) or late training (fine-tuning)

### Comparing Early vs Late Training

1. Set the left handle to iteration 0
2. Slowly drag the right handle from 0 to max
3. Watch how the solution and coefficients evolve as training progresses

---

## Data Requirements

The dashboard expects two files from a training run:

| File | Format | Required Content |
|------|--------|-----------------|
| `training_results.json` | JSON | `metadata` (benchmark_coefficients, ode_matrix), `milestones` (iteration, coefficients, objective) |
| `loss.csv` | CSV | `iteration,total,bc,pde,supervised` columns |

If `loss.csv` is missing, loss plots show a "(no loss CSV)" placeholder and the dashboard still works for solution/coefficient analysis.

---

## Troubleshooting

### Window Opens Then Closes Immediately

This is expected — the window stays open until you close it. If it closes immediately, check for errors in the terminal output.

### No Display / Headless Environment

GLMakie requires a display. On remote servers, use X11 forwarding (`ssh -X`) or a VNC session.

### Slow Startup

The first run compiles GLMakie and Makie (1–2 minutes). Subsequent launches are faster.

### Dark Mode Text Hard to Read

Try `--theme light` for a light-background variant, or `--theme high_contrast` for yellow-on-black.

---

*See also: [Viz.jl](../viz-modules/viz.md), [NNViewer.jl](../viz-modules/nnviewer.md), [Theme.jl](../viz-modules/theme.md)*
