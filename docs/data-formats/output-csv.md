# Output CSV Format

Loss history tracking during training.

**Location:** `<output_dir>/loss.csv`

---

## Structure

```csv
iteration,total,bc,pde,supervised
1,0.9500,0.3000,0.5000,0.1500
2,0.8200,0.2500,0.4200,0.1500
3,0.7100,0.1800,0.3500,0.1800
...
100000,0.0012,0.0001,0.0008,0.0003
```

---

## Columns

| Column | Description |
|--------|-------------|
| `iteration` | Training iteration number (1-based) |
| `total` | Optimized objective |
| `bc` | Boundary/initial-condition diagnostic loss |
| `pde` | ODE residual loss |
| `supervised` | Supervised loss |

---

## Reading in Julia

```julia
using CSV, DataFrames

df = CSV.read("loss.csv", DataFrame)

# Get total loss values
total_loss = df.total
```

---

## Reading in Python

```python
import pandas as pd

df = pd.read_csv("loss.csv")

# Get total loss values
total_loss = df["total"].values
```

---

## Plotting Loss Curves

```python
import matplotlib.pyplot as plt

plt.semilogy(df["iteration"], df["total"])
plt.xlabel('Iteration')
plt.ylabel('Total Loss')
plt.title('Training Loss')
plt.show()
```

---

## Analysis

Look for:
- **Monotonic decrease:** Healthy training
- **Plateaus:** May need more iterations or learning rate adjustment
- **Oscillations:** Learning rate too high
- **Diagnostic balance:** compare optimized losses with the BC/IC diagnostic

---

*See also: [Loss Components](../concepts/loss-components.md), [helper_funcs.jl](../julia-modules/helper-funcs.md)*
