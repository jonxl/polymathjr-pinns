# Tutorial: Scaling Experiments

Systematic studies of network size and training duration.

---

## Overview

This tutorial covers:
1. Neuron scaling experiments
2. Iteration scaling experiments
3. Combined analysis
4. Interpreting scaling curves

---

## Neuron Scaling

### Purpose

Determine optimal network width for your problem.

### Setup

```julia
include("utils/training_schemes.jl")

settings = TrainingSchemesSettings(
    training_dataset = load_training_data(),
    benchmark_dataset = load_benchmark_data(),
    N = 10,
    num_supervised = 5,
    num_points = 20,
    x_left = 0.0f0,
    x_right = 1.0f0,
    supervised_weight = 0.1f0,
    bc_weight = 1.0f0,
    pde_weight = 1.0f0,
    xs = collect(range(0, 1, 20))
)

neurons_counts = Dict(
    10 => "10_neurons",
    20 => "20_neurons",
    50 => "50_neurons",
    100 => "100_neurons",
    200 => "200_neurons"
)

scaling_neurons(settings, neurons_counts)
```

### Output

Creates separate directories:
```
data/scaling-neurons/
├── 10_neurons/
├── 20_neurons/
├── 50_neurons/
├── 100_neurons/
└── 200_neurons/
```

Each contains:
- `iteration_output.csv`
- Plots

---

## Iteration Scaling

### Purpose

Determine sufficient training duration.

### Setup

```julia
iteration_counts = Dict(
    500 => "500_iters",
    1000 => "1000_iters",
    5000 => "5000_iters",
    10000 => "10000_iters",
    50000 => "50000_iters"
)

run_training(settings, 10000, 100; neuron_count=100, seed=1234)
```

### Output

```
data/scaling-iters/
├── 500_iters/
├── 1000_iters/
├── 5000_iters/
├── 10000_iters/
└── 50000_iters/
```

---

## Analyzing Results

### Collect Final Losses

```julia
using CSV, DataFrames

function get_final_loss(dir)
    df = CSV.read(joinpath(dir, "iteration_output.csv"), DataFrame)
    row = df[df.loss_type .== "total_loss", :]
    return parse(Float64, names(row)[end])  # Last iteration
end

# Neuron scaling
neuron_losses = Dict()
for (n, name) in neurons_counts
    dir = joinpath("data/scaling-neurons", name)
    neuron_losses[n] = get_final_loss(dir)
end
```

### Plot Scaling Curve

```julia
using Plots

neurons = sort(collect(keys(neuron_losses)))
losses = [neuron_losses[n] for n in neurons]

plot(neurons, losses,
    xlabel="Number of Neurons",
    ylabel="Final Loss",
    yscale=:log10,
    marker=:circle,
    title="Neuron Scaling"
)
savefig("neuron_scaling.png")
```

---

## Interpreting Results

### Neuron Scaling

| Pattern | Interpretation |
|---------|----------------|
| Loss decreases then plateaus | Found sufficient capacity |
| Loss keeps decreasing | May need more neurons |
| Loss increases | Possible overfitting |

**Optimal:** Choose smallest network where loss plateaus.

### Iteration Scaling

| Pattern | Interpretation |
|---------|----------------|
| Loss decreases then plateaus | Sufficient training |
| Loss still decreasing | Need more iterations |
| Loss oscillates | Learning rate too high |

**Optimal:** Choose iteration count where loss stabilizes.

---

## Combined Grid

For thorough analysis, vary both:

```julia
for neurons in [20, 50, 100]
    for iters in [1000, 5000, 10000]
        name = "$(neurons)n_$(iters)i"
        # Configure and train
    end
end
```

### Heatmap Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

neurons = [20, 50, 100]
iters = [1000, 5000, 10000]
losses = np.array([
    [0.05, 0.02, 0.01],
    [0.03, 0.01, 0.005],
    [0.02, 0.008, 0.004]
])

plt.imshow(losses, cmap='viridis')
plt.xticks(range(len(iters)), iters)
plt.yticks(range(len(neurons)), neurons)
plt.xlabel('Iterations')
plt.ylabel('Neurons')
plt.colorbar(label='Final Loss')
plt.title('Scaling Grid')
plt.savefig('scaling_grid.png')
```

---

## Memory Considerations

| Neurons | Approximate Memory |
|---------|-------------------|
| 50 | ~100 MB |
| 100 | ~200 MB |
| 200 | ~500 MB |
| 500 | ~2 GB |

For large networks:
- Reduce batch size
- Use fewer collocation points
- Consider GPU acceleration

---

## Best Practices

1. **Start small:** 20 neurons, 1000 iterations
2. **Verify convergence:** Check loss curves
3. **Scale up:** Increase one parameter at a time
4. **Document:** Record all configurations

---

*See also: [training_schemes.jl](../julia-modules/training-schemes.md), [Hyperparameter Tuning](../concepts/hyperparameter-tuning.md)*
