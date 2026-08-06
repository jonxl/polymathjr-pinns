# Hyperparameter Tuning

Strategies for optimizing PINN training parameters.

---

## Key Hyperparameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `neuron_num` | Hidden layer width | 10 - 200 |
| `maxiters_adam` | Adam iterations | 1000 - 50000 |
| `n_terms_for_power_series` | Polynomial degree N | 5 - 20 |
| `num_supervised` | Supervised coefficients | 0 - N/2 |
| `num_points` | Collocation points | 10 - 100 |
| `supervised_weight` | Supervision strength | 0.0 - 1.0 |
| `pde_weight` | PDE loss weight | 0.1 - 10.0 |

---

## Tuning Strategies

### 1. Grid Search

Exhaustive search over a 2D grid of loss weights.

**File:** `utils/two_d_grid_search_hyperparameters.jl`

```julia
# Search over supervised_weight × pde_weight
weights = [(0.1, 0.5), (0.1, 1.0), (0.5, 0.5), ...]
```

**Output:** Contour plot of loss vs weight combinations

---

### 2. Binary Search

Efficient search for single weight optimization.

**File:** `utils/binary_search_on_weights.jl`

**Process:**
1. Start with bounds `[low, high]`
2. Evaluate midpoint
3. Narrow bounds based on loss
4. Repeat until convergence

---

### 3. Scaling Studies

Systematic exploration of network size and training duration.

**File:** `utils/training_schemes.jl`

**Neuron scaling:**
```julia
neurons = [10, 20, 50, 100, 200]
# Train separate model for each
```

**Iteration scaling:**
```julia
iters = [500, 1000, 5000, 10000, 50000]
# Train with each duration
```

---

## Recommended Workflow

1. **Start simple:** `neuron_num = 50`, `maxiters = 5000`
2. **Check convergence:** Loss should decrease steadily
3. **Scale neurons:** If underfitting, increase neurons
4. **Scale iterations:** If not converged, increase iterations
5. **Tune weights:** Grid search for optimal loss balance
6. **Validate:** Test on benchmark dataset

---

## Common Pitfalls

| Problem | Symptom | Solution |
|---------|---------|----------|
| Underfitting | High final loss | More neurons/iterations |
| Overfitting | Train << test loss | Less supervision, regularize |
| Slow convergence | Loss plateaus early | Adjust learning rate |
| Unstable training | Loss oscillates | Reduce learning rate |

---

## Analyzing Results

### Loss Curves

Look for:
- Monotonic decrease
- No sudden jumps
- Convergence (flattening)

### Coefficient Comparison

Check:
- Early coefficients match well
- Error grows for higher coefficients
- Boundary condition coefficients are accurate

---

*See also: [Training Schemes](../julia-modules/training-schemes.md), [Hyperparameter Search Tutorial](../tutorials/hyperparameter-search.md)*
