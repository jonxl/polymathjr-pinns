# PINN_RNN.jl

RNN-based PINN architecture using GRU cells.

**Location:** `architectures/PINN_RNN.jl`

---

## Overview

Alternative architecture that processes input as a sequence using recurrent layers.

---

## Architecture

```
Recurrence(GRUCell(1 → 64)) → Dense(64 → N+1)
```

Processes a sequence of inputs and outputs power series coefficients.

---

## When to Use

- Experimental: for research comparisons
- Sequential data patterns
- Alternative to feedforward when standard PINN underperforms

---

## Current Status

Experimental. The feedforward PINN (`PINN.jl`) is the primary implementation.

---

*See also: [PINN.jl](pinn.md)*
