# visualizer.py (deprecated)

The Python `nn-viewer` visualization tool has been replaced by a native
Julia + GLMakie dashboard in the `viz/` directory.

---

## Migration

| Python (old) | Julia (new) |
|---|---|
| `scripts/visualizer.py` | `viz/Viz.jl` |
| `scripts/main.py` | `src/explore.jl` |
| `python main.py --results ... --loss ...` | `julia --project src/explore.jl` |
| Matplotlib + PyQt5 backend | GLMakie (OpenGL) |
| `GeneralizedVisualizer` / `ODEResultsVisualizer` | `Viz.explore()` → `NNViewer.view()` |
| `PlotConfig` / `SliderConfig` dataclasses | GLMakie reactive observables |
| Dark / Light / High-contrast themes | `Viz.Theme` (dark / light / high_contrast) |
| Single range slider (matplotlib `RangeSlider`) | `GLMakie.IntervalSlider` |

---

## See Also

- [Visualization Guide](../tutorials/visualization-guide.md) — full tutorial for the new dashboard
- [Viz.jl](../viz-modules/viz.md) — dashboard module
- [NNViewer.jl](../viz-modules/nnviewer.md) — interactive viewer
- [Theme.jl](../viz-modules/theme.md) — colour themes
