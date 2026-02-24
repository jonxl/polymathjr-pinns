# Batched PINN Training (Single-Kernel GPU Parallelism)

## Status: Future Idea (not implementing now — went with threaded approach instead)

## The Idea

Instead of running B independent PINN training loops, stack B copies of network parameters into a batch dimension and run **one training loop** that trains all B PINNs simultaneously inside each iteration.

- One forward pass = one batched matmul computing B outputs
- One `Zygote.gradient()` call = B gradient sets
- One Adam step = updates all B parameter copies

Each of the B PINNs shares the same architecture and training data but uses **different loss weights** (its unique grid search point). Parameters diverge as training progresses.

## Why It's Interesting

The current grid search over `pde_weight x supervised_weight` can be 100x100 = 10,000 configurations. Each one triggers a full `train_pinn()` call. The GPU kernels for our tiny network (~2MB) are small and finish fast — the bottleneck is the **10,000 sequential kernel launches**, not single-kernel throughput.

Batching eliminates that. One kernel launch serves all B configs. The GPU's SIMT architecture is designed exactly for this — thousands of threads executing the same instruction on different data.

## How It Would Work

### Forward Pass
- Stack B parameter matrices: weights become `(out, in, B)` instead of `(out, in)`
- `Dense` layer forward pass becomes batched matmul: `Y = W_batched * X` where the batch dim flows through
- All B networks produce outputs simultaneously

### Loss Computation
- Each PINN in the batch gets its own `pde_weight` and `supervised_weight`
- Loss function returns a **vector of B losses**, not a scalar
- Each loss is independent — no reduction across batch

### Backward Pass
- `Zygote.gradient()` through the batched forward + loss
- Returns B independent gradient sets (one per config)

### Optimizer
- Adam state (m, v) also batched — `(param_size, B)` tensors
- Single vectorized Adam update step for all B configs

## Why We Didn't Do It (Yet)

| Concern | Detail |
|---------|--------|
| `Optimization.solve()` incompatible | Expects scalar loss, not a vector of B losses |
| Custom training loop required | Manual Adam loop (straightforward but more code) |
| Batched Lux forward pass | Need to reshape parameters so batch dim flows through `Dense` layers |
| Zygote through batched ops | Should work (bigger matmuls), but needs validation |
| Code surface area | Changes training loop + loss function + network forward pass |

The simpler **threaded approach** (spawning B independent `train_pinn()` calls via `Threads.@spawn`) achieves parallelism with changes only to `grid_search_2d()`. We went with that.

## VRAM Budget Estimate

- Network params: ~2MB per PINN
- Adam state (m + v): ~4MB per PINN
- AD tape overhead: ~10-20MB per PINN (estimate)
- Conservative: ~30MB per PINN instance
- 10GB VRAM -> ~300 concurrent configs
- 24GB VRAM -> ~800 concurrent configs

These numbers mean we could potentially batch an entire 100-point row or even the full grid in one shot.

## Comparison: Threaded vs Batched

| | Threaded (chosen) | Batched (this idea) |
|---|---|---|
| Training loops | B independent loops | 1 loop |
| `Optimization.solve()` | B separate calls | Can't use (vector of losses) |
| GPU kernel launches/iter | B x (kernels per PINN) | ~1 big kernel |
| Code changes | Only `grid_search_2d()` | Training loop + loss + forward pass |
| GPU efficiency | Good (stream interleaving) | Optimal (true SIMT batching) |
| Complexity | Low | High |

## When to Revisit

- If the threaded approach hits GPU utilization limits (streams not interleaving well)
- If we scale to much larger grids (1000x1000+) where kernel launch overhead dominates
- If we move to larger networks where batched matmul speedup is more pronounced

@claude-opus-4.6: This is the "do it right" approach for maximum GPU throughput. The threaded approach is the pragmatic first step — if it bottlenecks, this is the next level.
