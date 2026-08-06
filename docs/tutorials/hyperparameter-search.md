# Tutorial: Hyperparameter Search

Running grid and binary search for optimal training parameters.

---

## Overview

This tutorial covers:
1. Setting up grid search bounds
2. Running 2D grid search
3. Running binary search
4. Interpreting results

---

## Grid Search

### Setup

Edit search parameters in `utils/two_d_grid_search_hyperparameters.jl`:

```julia
# Define search grid
supervised_weights = [0.01, 0.05, 0.1, 0.5, 1.0]
pde_weights = [0.1, 0.5, 1.0, 2.0, 5.0]

# Fixed parameters
neuron_num = 50
maxiters = 5000
```

### Running

```julia
include("utils/two_d_grid_search_hyperparameters.jl")

# Run grid search
results = run_grid_search(
    training_data,
    benchmark_data,
    supervised_weights,
    pde_weights
)
```

### Output

Creates:
- `grid_search_results.csv` - All (weight, loss) combinations
- `grid_search_contour.png` - Contour plot visualization

---

## Binary Search

### Setup

```julia
include("utils/binary_search_on_weights.jl")

# Search bounds
low = 0.01
high = 1.0
target_weight = :supervised_weight  # or :pde_weight
tolerance = 0.001
```

### Running

```julia
optimal = binary_search_weight(
    training_data,
    benchmark_data,
    target_weight,
    low,
    high,
    tolerance
)
println("Optimal weight: ", optimal)
```

### Algorithm

```
1. mid = (low + high) / 2
2. Train with weight = mid
3. Evaluate loss
4. If loss_mid < loss_low: low = mid
   Else: high = mid
5. Repeat until |high - low| < tolerance
```

---

## Interpreting Results

### Contour Plot

![Grid Search Contour](../assets/grid_search_example.png)

- **Darker regions:** Lower loss (better)
- **Star marker:** Optimal point
- **Contour lines:** Loss level sets

### What to Look For

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| Single minimum | Clear optimal | Use those weights |
| Flat region | Insensitive to weight | Any value in region OK |
| Multiple minima | Complex landscape | Try different seeds |
| Edge minimum | Search range too narrow | Expand bounds |

---

## Recommended Workflow

1. **Coarse grid:** 5×5 with wide bounds
2. **Identify region:** Find promising area
3. **Fine grid:** 10×10 in promising region
4. **Binary search:** Refine individual weights

---

## Example: Full Search

```julia
# Step 1: Coarse search
coarse_sup = [0.01, 0.1, 0.5, 1.0, 2.0]
coarse_pde = [0.1, 0.5, 1.0, 2.0, 5.0]
coarse_results = run_grid_search(data, bench, coarse_sup, coarse_pde)

# Step 2: Identify best region (e.g., sup=0.1, pde=1.0)

# Step 3: Fine search around best
fine_sup = range(0.05, 0.2, length=10)
fine_pde = range(0.5, 1.5, length=10)
fine_results = run_grid_search(data, bench, fine_sup, fine_pde)

# Step 4: Binary refinement
optimal_sup = binary_search_weight(data, bench, :supervised_weight, 0.08, 0.15, 0.001)
```

---

## Performance Tips

- **Reduce iterations:** Use 1000-2000 for search, full training after
- **Smaller batch:** Use subset of ODEs for faster evaluation
- **Parallel search:** Run different seeds simultaneously

---

*See also: [Hyperparameter Tuning](../concepts/hyperparameter-tuning.md), [training_schemes.jl](../julia-modules/training-schemes.md)*
