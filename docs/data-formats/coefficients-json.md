# Coefficients JSON Format

Format for storing predicted coefficients for visualization.

---

## Structure

```json
{
  "10": [4.0, 1.95, 0.98, 1.30, ...],
  "20": [4.0, 1.99, 1.00, 1.32, ...],
  "50": [4.0, 2.00, 1.00, 1.33, ...],
  "100": [4.0, 2.00, 1.00, 1.33, ...]
}
```

---

## Schema

| Key | Value |
|-----|-------|
| Neuron count (string) | Array of predicted coefficients |

---

## Purpose

Used by the `Viz` diagnostic dashboard to:
- Compare predictions across training snapshots
- Enable iteration-range slider exploration
- Generate solution and coefficient comparison plots

The dashboard reads coefficients from `training_results.json` (`milestones[].coefficients` and `metadata.benchmark_coefficients`), not from a standalone coefficients file.

---

## Creating from Training Output

After running training with different neuron counts:

```julia
# Pseudo-code
results = Dict()
for neurons in [10, 20, 50, 100]
    settings = PINNSettings(neuron_num=neurons, ...)
    p_trained, net, st = train_pinn(settings, csv)
    coeffs = predict_coefficients(net, p_trained, st, input)
    results[string(neurons)] = coeffs
end
JSON.write("coefficients.json", results)
```

---

## Loading in Julia

```julia
using JSON

data = JSON.parsefile("training_results.json")
benchmark = data["metadata"]["benchmark_coefficients"]

for m in data["milestones"]
    it = m["iteration"]
    coeffs = m["coefficients"]
    # ...
end
```

---

## Integration with Dashboard

```bash
julia --project src/explore.jl results/run-adam-07-23-26/training_results.json results/run-adam-07-23-26/loss.csv
```

Or from the REPL:

```julia
include("viz/Viz.jl")
using .Viz
Viz.explore("results/run-adam-07-23-26/training_results.json",
            "results/run-adam-07-23-26/loss.csv")
```

---

*See also: [Viz.jl](../viz-modules/viz.md), [Visualization Guide](../tutorials/visualization-guide.md)*
