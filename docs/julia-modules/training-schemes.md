# training_schemes.jl

Training strategies for hyperparameter exploration.

**Location:** `utils/training_schemes.jl`

---

## TrainingSchemesSettings Struct

```julia
struct TrainingSchemesSettings
    training_dataset::Dict{String,Dict{String,Any}}
    benchmark_dataset::Dict{String,Dict{String,Any}}
    N::Int                      # Power series degree
    num_supervised::Int
    num_points::Int
    x_left::Float32
    x_right::Float32
    supervised_weight::Float32
    pde_weight::Float32
    xs::Vector{Float32}
end
```

---

## Training Functions

### `scaling_neurons(settings, neurons_counts)`

Trains separate PINNs with different neuron counts.

```julia
scaling_neurons(settings, neurons_counts::Dict)
```

**Example:**
```julia
neurons_counts = Dict(10 => "small", 50 => "medium", 100 => "large")
scaling_neurons(settings, neurons_counts)
```

Creates separate training runs for each neuron count.

---

### `run_training(settings, maxiters, milestone_interval; kwargs...)`

Unified training path — trains a PINN once, evaluates at milestones.

```julia
run_training(settings::TrainingSchemesSettings, maxiters::Int, milestone_interval::Int;
             snapshot_path=nothing, batch_size=0, snapshot_epoch_interval=10,
             neuron_count=100, seed=1234, representation=:power_series)
```

**Example:**
```julia
run_training(settings, 10000, 100;
             batch_size=32, neuron_count=100, seed=1234)
```

Writes `model.safetensors` for the final trained MLP, plus `training_results.json` with metadata, final results, and checkpoint history. Intermediate checkpoint weights are written under `snapshots/` when checkpointing is enabled.

---

### `grid_search_at_scale(settings, neurons_counts)`

2D grid search over the optimized loss weights at different network scales.

```julia
grid_search_at_scale(settings, neurons_counts::Dict)
```

Combines neuron scaling with hyperparameter grid search.

---

## Use Cases

| Function | Use Case |
|----------|----------|
| `scaling_neurons` | Find optimal network size |
| `run_training` | Train a PINN with configurable hyperparameters and optional milestone snapshots |
| `grid_search_at_scale` | Full hyperparameter optimization |

---

*See also: [Hyperparameter Tuning](../concepts/hyperparameter-tuning.md), [Scaling Experiments Tutorial](../tutorials/scaling-experiments.md)*
