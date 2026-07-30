# NNViewer.jl

Interactive PINN ODE training-results viewer backed by GLMakie.

**Location:** `viz/NNViewer.jl`

---

## Entry Point

```julia
NNViewer.view(results_json_path::String, loss_csv_path::String;
              theme          = nothing,
              x_range        = (0.0, 1.0),
              num_points     = 1000,
              initial_start  = nothing,
              initial_end    = 1000)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results_json_path` | `String` | *(required)* | Path to `training_results.json` |
| `loss_csv_path` | `String` | *(required)* | Path to `loss.csv` |
| `theme` | `Theme` | `nothing` | Theme from `Theme.jl` |
| `x_range` | `Tuple` | `(0.0, 1.0)` | Evaluation domain |
| `num_points` | `Int` | `1000` | Number of x collocation points |
| `initial_start` | `Int` | `iter_min` | Initial left slider handle |
| `initial_end` | `Int` | `1000` | Initial right slider handle |

---

## Dashboard Layout

8 diagnostic panels (4×2 grid):

### Row 3: Solution Analysis
- **ODE Solution Comparison** — benchmark (cyan dashed) vs PINN (orange solid)
- **Absolute Error of Solution** — log-scale `|benchmark - PINN|` (red)

### Row 4: Coefficient Analysis
- **Coefficient Comparison** — benchmark (cyan) vs PINN (orange)
- **Coefficient Error** — log-scale per-coefficient error (red)

### Rows 5–6: Loss Curves
- **Total Loss** (grey), **BC Loss** (red), **PDE Loss** (blue), **Supervised Loss** (green)
- Full training history with yellow span highlight between slider handles
- Dashed vline at start handle, solid vline at end handle

### Controls
- **Row 8:** `GLMakie.IntervalSlider` — single bar, two drag handles
- **Row 7:** Iteration range readout
- **Row 9:** Status bar (snapshot iteration + objective)

---

## Data Series Colours

| Series | Hex |
|--------|-----|
| Benchmark | `#4fc3f7` (cyan) |
| PINN | `#ff8a65` (orange) |
| Analytical | `#66bb6a` (green) |
| Error | `#ef5350` (red) |
| Total Loss | `#e0e0e0` |
| BC Loss | `#ef5350` |
| PDE Loss | `#42a5f5` |
| Supervised Loss | `#66bb6a` |
| Marker | `#ffeb3b` (yellow) |

---

## Internal Helpers

| Function | Description |
|----------|-------------|
| `evaluate_factorial_power_series(coeffs, x)` | Evaluates `Σ c_n · xⁿ / n!` with incremental factorial |
| `style_axis!(ax, theme)` | Applies theme colours to axis background, grid, labels, ticks |

---

*See also: [Viz.jl](viz.md), [Theme.jl](theme.md), [Visualization Guide](../tutorials/visualization-guide.md)*
