# main.py Examples (deprecated)

The Python visualization tools have been replaced by a native Julia + GLMakie
dashboard.  See [Visualization Guide](../tutorials/visualization-guide.md).

---

## Migration

| Python (old) | Julia (new) |
|---|---|
| `python main.py` | `julia --project src/explore.jl` |
| `python main.py --results ... --loss ...` | `julia --project src/explore.jl results/.../training_results.json results/.../loss.csv` |
| `setup_backend()` | (GLMakie handles backends automatically) |
| `PowerSeriesVisualizer(...)` | `Viz.explore(results_json, loss_csv; theme=...)` |

---

*See also: [Viz.jl](../viz-modules/viz.md), [NNViewer.jl](../viz-modules/nnviewer.md)*
