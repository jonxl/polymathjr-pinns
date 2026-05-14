# Mini-Batch Training for global_loss

**Status:** Open
**Priority:** High
**Created:** February 25, 2026

## Context

@architect: Training on 372 ODEs with SCALING_ADAM mode. Loss dropped from 8.653e7 → 8.367e7 over 6700 Adam iterations — only 3.3% reduction. Linear decay, not exponential. Process killed by SIGKILL before completing 10,000 iterations.

@claude-opus-4.6: Diagnosed two root causes. First, the SIGKILL is an OOM kill — system has 15 GB RAM with zero swap, GPU VRAM is 11.2 GB free (not the bottleneck). The Zygote AD tape for 372 ODEs in `global_loss` exhausts system RAM. Second, the slow convergence comes from catastrophic scale mismatch in training data — ODE `[-4, 1]` has coefficients up to 524,288 while ODE `[1, -4]` maxes at 4. The supervised MSE loss gives the large-coefficient ODE 17 billion times more gradient weight. Proposed three fixes: relative MSE normalization, per-ODE normalization, or mini-batching.

@architect: The normalization approach is wrong for this project — the PINN would optimize to a log-scale version of the coefficients to keep the loss lower. The network needs to learn actual coefficient values, not relative ones. Also the more immediate issue is the SIGKILL crash.

@claude-opus-4.6: Agreed — normalization changes what the network optimizes for, which is undesirable when downstream tools (nn-viewer, benchmark evaluation) care about absolute coefficient values. Mini-batching addresses both problems without changing the loss function: reduces memory (fixes OOM) and introduces stochastic gradient noise (helps convergence). The network still optimizes the same total loss — just via unbiased stochastic estimates.

## Definitions

These terms apply throughout this plan and the implementation:

**Iteration** — One call to `loss_wrapper` → one Zygote AD tape → one gradient computation → one weight update by the optimizer. This is the atomic unit of training. Currently (`PINN.jl:L298-L353`), every iteration processes ALL training ODEs. With mini-batching, each iteration processes only one bin.

**Bin** — A subset of the training ODEs processed in a single iteration. With 372 ODEs and `batch_size=32`, the data is partitioned into bins:

```
Bin  1: ODEs  1–32   → loss → gradient → update weights
Bin  2: ODEs 33–64   → loss → gradient → update weights (using updated weights from bin 1)
...
Bin 11: ODEs 321–352 → loss → gradient → update weights
Bin 12: ODEs 353–372 → loss → gradient → update weights (20 leftover ODEs — smaller bin)
```

Each bin = one iteration = one Zygote tape. The tape is created when `loss_wrapper` is called and garbage-collected when it returns, before the next bin starts. This is what keeps memory bounded.

**Epoch** — One complete pass through ALL training ODEs. With 372 ODEs and `batch_size=32`, one epoch = 12 bins (11 bins of 32 + 1 bin of 20 leftovers). After an epoch completes, the 372 ODEs are re-shuffled into a new random order and partitioned into fresh bins. This means bin 1 in epoch 2 contains different ODEs than bin 1 in epoch 1.

**Relationship:** With 10,000 total iterations and 12 bins per epoch: 10,000 iterations = ~833 epochs. Every ODE gets seen ~833 times over the full training run.

**Snapshot timing:** Snapshots fire at **epoch boundaries** only — after all bins in an epoch have been processed and every ODE has been seen at least once since the last snapshot. Never mid-epoch. The interval is configurable: `SNAPSHOT_EVERY_N_EPOCHS` controls how many epochs pass between snapshots.

## Problem

Two interrelated failures when training on 372 ODEs simultaneously:

### 1. OOM crash — SIGKILL after ~6700 iterations

**System:** 15 GB RAM, **zero swap**, RTX 3060 (12 GB VRAM — largely unused).

`global_loss` (`architectures/PINN.jl:L210-L243`) loops over **all 372 ODEs** inside a single Zygote-differentiated call. The AD tape accumulates intermediate state from 372 forward passes through the network before the backward pass can release any of it. With 15 GB RAM and no swap, Julia's memory usage eventually triggers the Linux OOM killer → `SIGKILL`.

Evidence: GPU VRAM is 11.2 GB free at idle. System RAM is 3.7 GB available. The crash is system RAM, not GPU memory.

### 2. Near-zero convergence — 3.3% loss reduction over 6700 iterations

Training output shows loss dropping from 8.653e7 → 8.367e7 over 6700 Adam iterations (lr=0.001). That's a **3.3% reduction** with a linear (not exponential) decay profile — classic sign of an optimizer stuck on a flat loss landscape.

**Root cause:** Catastrophic scale mismatch in supervised targets.

@architect: I really would not worry about this because this is part fo the research problem and how this is how the problem is setup. In a sense this has to do with the TYPE of loss function. This makes me want you to turn your attention to HOW we load up which optimizer we use. We want to continue with more optimizers that Lux natively has so that we can replace the optimizer with something else for our research.

@claude-opus-4.6: Understood — the scale mismatch is a known property of the problem, not a bug to fix. Keeping the convergence analysis below for reference but deprioritizing it. On optimizers: currently the optimizer is hardcoded as `Adam(0.001f0)` at `PINN.jl:L346`, and the `PINNSettings.optimizer` field (`PINN.jl:L79`) is a string that's only used for naming run directories — it never actually selects the optimizer. Making this configurable is a separate plan item. The project already depends on `OptimizationOptimisers` (which wraps Optimisers.jl, not Lux directly) so all Optimisers.jl algorithms (AdamW, RAdam, NAdam, Lion, etc.) are available. Will create a separate plan for optimizer swapping.

The training data contains ODEs whose power series coefficients span 8+ orders of magnitude:

| ODE (ay + by' = 0) | Eigenvalue | Coefficient range | Max |
|---------------------|-----------|-------------------|-----|
| `[-4, 1]` (y'=4y) | +4 | 2 → 524,288 | 524,288 |
| `[9, -5]` | +5/9 | 4 → 793 | 793 |
| `[1, -4]` (y'=y/4) | +1/4 | 4 → 0.00002 | 4 |
| `[-5, -9]` | -9/5 | 1 → -0.005 | 1 |

The supervised loss (`utils/loss_functions.jl:L219-L220`) uses raw MSE:

```julia
diff = (settings.a_vec - padded_data) .* mask
return sum(abs2, diff) / K
```

A single ODE like `[-4, 1]` (max coefficient 524,288) contributes ~(524288)^2/10 ≈ **2.7e10** to the supervised loss. An ODE like `[1, -4]` contributes ~(4)^2/10 ≈ **1.6**. That is a **17-billion-to-one** ratio. A handful of ODEs with large positive eigenvalues dominate the entire gradient — the network spends all its capacity trying (and failing) to output values of ~500k.

### Why NOT loss normalization?

Per-ODE normalization (relative MSE, log-scale loss) would equalize gradient contributions, but it changes **what the network optimizes for**. The PINN would learn to minimize relative error on a log scale rather than absolute coefficient accuracy. This is undesirable when the downstream use (nn-viewer comparison, benchmark evaluation) cares about actual coefficient values.

## Solution — Mini-Batching

Instead of computing `global_loss` over all 372 ODEs per iteration, partition the ODEs into **bins** and process one bin per iteration. After all bins are processed (one **epoch**), re-shuffle and repeat.

### Why this works

1. **Fixes OOM:** 32 ODEs on the Zygote tape instead of 372 = ~12x less peak memory.
2. **Improves convergence:** Standard SGD insight — stochastic gradient noise helps escape flat regions and saddle points. Each iteration is ~12x faster, yielding more parameter updates per minute.

@architect: I think I see what you are saying. So we run the PINN on one bin at a time which allows the PINN to optimize for that bin then we carry the weights over to be trained on the next bin?

@claude-opus-4.6: Exactly right, but with one important detail: we don't just train on each bin once and move on — we **cycle through all bins repeatedly**. Here's the flow for 372 ODEs with batch_size=32:

**One epoch** (one full pass through all training data):
1. Shuffle all 372 ODEs randomly
2. Split into 12 bins: bins 1–11 have 32 ODEs each (352 total), bin 12 has the remaining 20
3. Train on bin 1 → compute gradient → update weights
4. Train on bin 2 (with updated weights) → compute gradient → update weights
5. ... continue through all 12 bins
6. That's 12 gradient updates per epoch, each seeing different ODEs

**Next epoch**: re-shuffle, split into new bins, repeat. Over many epochs every ODE contributes equally to training — the bins just control how many ODEs are in memory at once.

With 10,000 total iterations and 12 bins per epoch, that's ~833 epochs. Every ODE gets seen ~833 times. The weights are never "stuck" optimizing for one bin — they continuously improve across all ODEs.

3. **Mitigates scale mismatch:** Any given mini-batch may not contain the worst outlier ODEs, so the gradient is not always dominated by the same handful of extreme examples. Over many iterations, all ODEs contribute to the gradient — but the per-step signal is less distorted.
4. **Preserves the loss function:** The network still optimizes the same total loss over all ODEs — just via unbiased stochastic estimates. No normalization, no log-scale, no change to what the network learns.

### Batch size choice

With 372 training examples:
- **32** — ~12x memory reduction, strong stochastic noise, standard for NN training
- **64** — ~6x memory reduction, moderate noise, slightly more stable gradients

Recommend starting with **32**, adjustable via configuration.

@architect: Yes this needs to be configurable. Also make sure to ensure that any left overs that do not fit cleanly into n FULL bins we take into account that they must be passed over to the next bin that can be filled. I just have a quesiont if we do one bin how is the memory "cleared" for the next bin?

@claude-opus-4.6: Two questions here:

**Leftovers:** 372 ODEs / 32 per bin = 11 full bins (352 ODEs) + 20 remaining. Those 20 go into a 12th smaller bin that gets processed normally. No ODEs are dropped. On the next epoch we re-shuffle, so those 20 ODEs will land in different bins — over many epochs the distribution evens out. The implementation handles this naturally: `randperm(372)` gives a shuffled index list, we slice it into chunks of 32, and the last chunk is simply shorter.

**Memory clearing:** This is where the Optimization.jl + Zygote architecture works in our favor. Each call to `loss_wrapper` (which calls `global_loss`) creates a fresh Zygote AD tape for only the ODEs in that batch. When `loss_wrapper` returns, the solver has the gradient and the loss value — the tape and all its intermediate allocations become unreachable. Julia's garbage collector then reclaims that memory before the next call. We're not doing anything special to clear memory — the natural call boundary between iterations means the tape from bin 1 is gone before bin 2 starts. That's the whole point: instead of one massive tape for 372 ODEs, we get 12 small tapes that each live only for the duration of one gradient computation.

To be safe, we can add an explicit `GC.gc(false)` (non-full GC) between epochs if memory pressure is still high, but this likely won't be necessary.

@architect: Snapshots should happen at epoch boundaries — after all bins have been processed and every ODE has been seen. Not mid-epoch.

@claude-opus-4.6: Agreed. Currently snapshots fire every N iterations (`PINN.jl:L323-L326`), which with batching could land in the middle of an epoch when only some bins have been processed. The fix: the `EpochBatchIterator` tracks epoch count, and `custom_callback` checks for epoch completion instead of `iteration % milestone_interval`. With ~833 epochs over 10,000 iterations, `SNAPSHOT_EVERY_N_EPOCHS` controls frequency (e.g., 10 = ~83 snapshots).

## Implementation Steps

### Step 1: Create `EpochBatchIterator` with epoch tracking

**File:** `architectures/PINN.jl`

The batch iterator is a mutable object that tracks the shuffled order, current position within an epoch, and epoch count. It signals when an epoch boundary is crossed so the callback can trigger snapshots at the right time.

```julia
mutable struct EpochBatchIterator
  all_items::Vector{Pair}    # all (matrix_key => series_coeffs) pairs
  order::Vector{Int}         # shuffled indices for current epoch
  pos::Int                   # current position in the epoch
  batch_size::Int            # ODEs per bin (0 = full batch)
  epoch_count::Int           # how many complete epochs have finished
  epoch_just_completed::Bool # true on the first call after an epoch wraps
end

function EpochBatchIterator(ode_matrices::Dict, batch_size::Int)
  items = collect(ode_matrices)
  order = batch_size > 0 ? randperm(length(items)) : collect(1:length(items))
  EpochBatchIterator(items, order, 1, batch_size, 0, false)
end

function next_batch!(iter::EpochBatchIterator)
  iter.epoch_just_completed = false
  n = length(iter.all_items)
  if iter.batch_size <= 0 || iter.batch_size >= n
    # Full batch mode — every call is a complete "epoch"
    iter.epoch_count += 1
    iter.epoch_just_completed = true
    return iter.all_items
  end
  # If we've exhausted this epoch, re-shuffle and start a new one
  if iter.pos > n
    iter.epoch_count += 1
    iter.epoch_just_completed = true
    iter.order = randperm(n)
    iter.pos = 1
  end
  # Take next chunk (may be smaller than batch_size for the last bin)
  batch_end = min(iter.pos + iter.batch_size - 1, n)
  indices = iter.order[iter.pos:batch_end]
  iter.pos = batch_end + 1
  return iter.all_items[indices]
end
```

Note: `epoch_just_completed` is set to `true` at the START of the first bin of a new epoch (i.e., when `next_batch!` detects the previous epoch exhausted all ODEs). This means the snapshot fires right after the last bin of the completed epoch finishes its gradient update, before the first bin of the next epoch runs.

### Step 2: Modify `global_loss` to accept a pre-selected batch

**File:** `architectures/PINN.jl`, function at L210

- Add `ode_items` kwarg — the pre-selected batch of ODE pairs for this iteration
- The caller (`loss_wrapper`) calls `next_batch!` inside `Zygote.ignore()` and passes the result
- Normalize by `length(ode_items)` instead of `length(settings.ode_matrices)`

```julia
function global_loss(p_net, settings::PINNSettings, coeff_net, st, use_gpu::Bool=false;
                     all_ode_buffers::Union{Dict, Nothing}=nothing,
                     ode_items::Union{Vector, Nothing}=nothing)

  # Use provided batch or fall back to full dataset
  items = ode_items !== nothing ? ode_items : collect(settings.ode_matrices)

  total_loss = F(0.0)
  ...
  num_in_batch = length(items)

  for (alpha_matrix_key, series_coeffs) in items
    ...  # unchanged inner loop
  end

  normalized_loss = total_loss / num_in_batch
  ...
end
```

### Step 3: Wire `EpochBatchIterator` through `loss_wrapper` in `train_pinn`

**File:** `architectures/PINN.jl`, function at L253

- Add `batch_size::Int=0` kwarg to `train_pinn`
- Create `EpochBatchIterator` before the optimization loop (after `precompute_buffers`)
- In `loss_wrapper`, call `next_batch!` inside `Zygote.ignore()` and pass to `global_loss`

```julia
function train_pinn(settings::PINNSettings, output_dir;
                    ...,
                    batch_size::Int=0,
                    snapshot_epoch_interval::Int=10)
  ...
  # Create batch iterator (after precompute_buffers, before optimization loop)
  batch_iter = EpochBatchIterator(settings.ode_matrices, batch_size)

  function loss_wrapper(p_net, _)
    # Batch selection is constant w.r.t. p_net — keep off AD tape
    items = Zygote.ignore() do
      next_batch!(batch_iter)
    end
    loss, losses = global_loss(p_net, settings, coeff_net, st, use_gpu;
                               all_ode_buffers=all_ode_buffers,
                               ode_items=items)
    latest_metrics[] = (losses.bc, losses.pde, losses.sup)
    return loss
  end
  ...
end
```

### Step 4: Change `custom_callback` to snapshot at epoch boundaries

**File:** `architectures/PINN.jl`, `custom_callback` at L304-L329

Currently, milestones fire when `iteration % milestone_interval == 0`. Replace with epoch-boundary detection using `batch_iter.epoch_just_completed`.

```julia
function custom_callback(state, l; progress_bar)
  iter_count[] += 1
  iteration = iter_count[]
  latest_params[] = state.u

  # Log metrics every LOG_INTERVAL iterations (unchanged)
  if iteration % LOG_INTERVAL == 0 || iteration == 1
    bc, pde, sup = latest_metrics[]
    idx = history_len[] + 1
    history_len[] = idx
    history_iters[idx] = iteration
    history_buf[idx, 1] = Float32(l)
    history_buf[idx, 2] = Float32(bc)
    history_buf[idx, 3] = Float32(pde)
    history_buf[idx, 4] = Float32(sup)
    progress_bar(state, l)
  end

  # Snapshot at epoch boundaries (replaces iteration % milestone_interval)
  epoch_done = Zygote.ignore() do
    batch_iter.epoch_just_completed
  end
  if on_milestone !== nothing && epoch_done &&
     snapshot_epoch_interval > 0 &&
     batch_iter.epoch_count % snapshot_epoch_interval == 0
    p_current = use_gpu ? ComponentArray(Array(getdata(state.u)), getaxes(state.u)) : state.u
    on_milestone(p_current, iteration, coeff_net, st, run_id)
  end

  return false
end
```

In full-batch mode (`batch_size=0`), every call to `next_batch!` sets `epoch_just_completed = true`, so this degrades to snapshotting every `snapshot_epoch_interval` iterations — same as the old `milestone_interval` behavior.

### Step 5: Wire `batch_size` and `snapshot_epoch_interval` through `scaling_adam`

**File:** `utils/training_schemes.jl`, function at L56

- Add `batch_size::Int=0` and `snapshot_epoch_interval::Int=10` kwargs to `scaling_adam`
- Pass both to `train_pinn`

```julia
function scaling_adam(settings::TrainingSchemesSettings, maxiters::Int,
                     milestone_interval::Int;
                     snapshot_path::Union{String,Nothing}=nothing,
                     batch_size::Int=0,
                     snapshot_epoch_interval::Int=10)
  ...
  p_trained, coeff_net, st, _, history = train_pinn(pinn_settings, output_dir;
      ...,
      batch_size=batch_size,
      snapshot_epoch_interval=snapshot_epoch_interval)
  ...
end
```

Note: `milestone_interval` is kept in the function signature for backward compatibility but is no longer used for snapshot timing when `batch_size > 0`. The `snapshot_epoch_interval` takes over.

### Step 6: Expose configuration in `src/main.jl`

**File:** `src/main.jl`, configuration section and SCALING_ADAM block

```julia
# =========================================================================
# Configuration: Mini-Batching
# =========================================================================
BATCH_SIZE = 32               # ODEs per bin. 0 = full batch (all ODEs per iteration)
SNAPSHOT_EVERY_N_EPOCHS = 10  # Save snapshot after every N complete epochs

# In SCALING_ADAM block:
scaling_adam(scaling_adam_settings, maxiters, milestone_interval;
            snapshot_path=snap,
            batch_size=BATCH_SIZE,
            snapshot_epoch_interval=SNAPSHOT_EVERY_N_EPOCHS)
```

With 372 ODEs, batch_size=32 → 12 bins/epoch → 10,000 iterations ≈ 833 epochs.
`SNAPSHOT_EVERY_N_EPOCHS = 10` → ~83 snapshots over the full run.

### Step 7: Ensure `evaluate_solution` stays full-batch

**File:** `architectures/PINN.jl`, function at L418

No changes needed. `evaluate_solution` calls `loss_fn` directly per benchmark ODE — it never uses `global_loss`. Benchmark evaluation is always full-batch.

## Files to Modify

| File | Changes |
|------|---------|
| `architectures/PINN.jl` | Add `EpochBatchIterator` struct + `next_batch!`. Add `ode_items` kwarg to `global_loss`. Add `batch_size` + `snapshot_epoch_interval` kwargs to `train_pinn`. Create iterator in `train_pinn`, call `next_batch!` in `loss_wrapper`. Change `custom_callback` milestone check from iteration-based to epoch-based. |
| `utils/training_schemes.jl` | Add `batch_size` + `snapshot_epoch_interval` kwargs to `scaling_adam`, pass through to `train_pinn`. |
| `src/main.jl` | Add `BATCH_SIZE` and `SNAPSHOT_EVERY_N_EPOCHS` config variables, pass to `scaling_adam`. |

## What Does NOT Change

- `loss_fn()` — untouched (per-ODE loss computation)
- `loss_functions.jl` — untouched (PDE/BC/supervised loss components)
- `precompute_buffers()` — still pre-computes all ODE buffers (small constant arrays, not the memory problem)
- `evaluate_solution()` — always full-batch on benchmark data
- `on_milestone` callback in `scaling_adam` — still calls `evaluate_solution` on benchmark data (full-batch)
- Snapshot file format — unchanged (.bin files with Float32 weights)
- Snapshot loading / warm-start — unaffected
- Grid search / SINGLE_RUN modes — default `batch_size=0` preserves full-batch behavior
- Adam learning rate (0.001) — unchanged; can be tuned separately

## Backward Compatibility

| Mode | `batch_size` | Behavior |
|------|-------------|----------|
| SCALING_ADAM (new) | 32 | Epoch-based batching, snapshots at epoch boundaries |
| SCALING_ADAM (legacy) | 0 | Full batch, every iteration = epoch, `snapshot_epoch_interval` acts like old `milestone_interval` |
| SINGLE_RUN | 0 (default) | Unchanged — no batching, no epoch tracking |
| GRID_SEARCH | 0 (default) | Unchanged — no batching |

## Verification

1. Run with `BATCH_SIZE = 32`, `maxiters = 10000`, monitor RAM via `htop`
   - Expect: no SIGKILL, peak RAM stays well under 15 GB
2. Verify epoch-based snapshots:
   - Expect: snapshots appear at epoch boundaries (every ~120 iterations with `SNAPSHOT_EVERY_N_EPOCHS=10`), NOT at fixed iteration counts
3. Verify all ODEs processed per epoch:
   - Add a one-time log at epoch boundary: `@info "Epoch $(batch_iter.epoch_count) complete ($(length(batch_iter.all_items)) ODEs processed)"`
4. Run with `BATCH_SIZE = 0` (full batch) on a small dataset:
   - Expect: identical behavior to current code (backward compatibility)
5. Check milestone evaluations are still full-batch on benchmark data:
   - Expect: same benchmark error values regardless of batch_size

## Future Considerations

- **Configurable optimizer (separate plan):** Currently `Adam(0.001f0)` is hardcoded at `PINN.jl:L346` and `PINNSettings.optimizer` (`PINN.jl:L79`) is a display-only string. Per @architect, this should be made configurable so the project can research different optimizers. The `OptimizationOptimisers` dependency already provides AdamW, RAdam, NAdam, Lion, etc. — just need a dispatch mechanism from settings to optimizer constructor. This is a separate plan item.
- **Learning rate schedule:** With mini-batching, higher learning rates (0.003–0.01) often work better. Could add lr as a configurable parameter.
- **Batch size warmup:** Start with small batches (high noise) and increase batch size during training — equivalent to learning rate annealing.
- **Data filtering:** Consider excluding ODEs with eigenvalues > threshold (e.g., |eigenvalue| > 5) whose coefficients grow too fast for a truncated power series to represent well.
