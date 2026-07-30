# Viz.jl

Top-level dashboard module.  Ties `Theme.jl` and `NNViewer.jl` together.

**Location:** `viz/Viz.jl`

---

## Module Structure

```julia
module Viz

include("Theme.jl")
using .Theme

include("NNViewer.jl")
using .NNViewer

function explore(results_json::String, loss_csv::String;
                 theme::Theme.Theme = Theme.DARK_THEME, kwargs...)
    NNViewer.view(results_json, loss_csv; theme=theme, kwargs...)
end

end # module Viz
```

---

## Exports

| Symbol | Type | Description |
|--------|------|-------------|
| `explore` | function | Launch interactive dashboard |
| `Theme` | struct | Colour scheme (re-exported from Theme.jl) |
| `DARK_THEME` | `Theme` | Dark theme (default) |
| `LIGHT_THEME` | `Theme` | Light theme |
| `HIGH_CONTRAST_THEME` | `Theme` | High-contrast theme |
| `get_theme` | function | Look up a theme by name |
| `list_themes` | function | List all registered theme names |

---

## Usage

### From the CLI

```bash
julia --project src/explore.jl
```

### From the REPL

```julia
include("viz/Viz.jl")
using .Viz

Viz.explore("results/run-adam-06-03-26/training_results.json",
            "results/run-adam-06-03-26/loss.csv")

# With a non-default theme
Viz.explore("results/run-adam-06-03-26/training_results.json",
            "results/run-adam-06-03-26/loss.csv";
            theme=Viz.LIGHT_THEME)
```

---

*See also: [NNViewer.jl](nnviewer.md), [Theme.jl](theme.md), [Visualization Guide](../tutorials/visualization-guide.md)*
