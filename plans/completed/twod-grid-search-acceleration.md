# Plan: Threaded 2D Grid Search Parallelization

## Context

The 2D grid search in `grid_search_2d()` runs a **sequential nested loop** over `num_points × num_points` weight configurations. For a 100×100 grid, that's 10,000 sequential `train_pinn()` calls. Each call is fully independent — different loss weights, same architecture, same data. We want to run multiple PINN instances concurrently using Julia threads, batched by VRAM capacity.

@architect: Good work on this implementation. You understood the context of the problem. The goal is to somehow run many configurations from the grid at once.

@claude-opus-4.6: Thanks. The nice thing is the existing code already isolates each config into its own directory and its own `PINNSettings` — so the parallelism slots in at the orchestration layer without touching the training internals.

## Key Insight: Already Thread-Safe (Mostly)

Each grid point gets a **unique `config_dir`** based on its weight values (e.g., `s0.3-b1.0-p0.7/`). So file I/O doesn't collide between grid points. The only shared mutable state is `best_objective` / `best_weights` tracking in the outer loop — easily fixed with post-hoc reduction.

## Files to Modify

| File | Change |
|------|--------|
| `utils/two_d_grid_search_hyperparameters.jl` | Parallelize `grid_search_2d()` inner loop with `Threads.@spawn`, add VRAM batching |

## Implementation Steps

### Step 1: Add thread-safe grid search function

In `grid_search_2d()` (line 118-143), replace the sequential nested loop:

```julia
# BEFORE (sequential):
for (i, w1) in enumerate(weight1_values)
  for (j, w2) in enumerate(weight2_values)
    objective_value = evaluate_weight_configuration(...)
    objective_matrix[j, i] = objective_value
    if objective_value < best_objective ...
  end
end

# AFTER (threaded batches):
all_configs = [(i, w1, j, w2) for (i, w1) in enumerate(weight1_values)
                                for (j, w2) in enumerate(weight2_values)]

batch_size = min(length(all_configs), estimate_batch_size())

for batch in Iterators.partition(all_configs, batch_size)
  tasks = map(batch) do (i, w1, j, w2)
    Threads.@spawn begin
      weights = create_weight_tuple(weight1, w1, weight2, w2, fixed_weights)
      evaluate_weight_configuration(neuron_count, training_dataset,
        benchmark_dataset, weights, num_supervised, N,
        x_left, x_right, xs, base_data_dir)
    end
  end

  # Collect results from this batch
  for (k, task) in enumerate(tasks)
    i, w1, j, w2 = batch[k]
    objective_matrix[j, i] = fetch(task)
  end
end

# Find best AFTER all batches (no locks needed)
best_idx = argmin(objective_matrix)
best_objective = objective_matrix[best_idx]
```

### Step 2: VRAM budget estimation

Add a helper function to estimate how many concurrent PINNs fit in VRAM:

```julia
function estimate_batch_size()
  if !GPUUtils.is_gpu_available()
    return Threads.nthreads()  # CPU-only: just use available threads
  end

  available_vram = CUDA.available_memory()
  per_pinn_estimate = 50 * 1024 * 1024  # ~50MB conservative (params + optimizer + AD tape)
  gpu_batch = max(1, Int(floor(available_vram / per_pinn_estimate)))

  # Don't exceed thread count — no point spawning more tasks than threads
  return min(gpu_batch, Threads.nthreads())
end
```

The 50MB estimate is conservative for our tiny network. Can be tuned after profiling. @architect: I would like to hear your comments on we could do profiling.

@claude-opus-4.6: Two practical approaches. (1) **Empirical**: run a single `train_pinn()` call and check VRAM before/after with `CUDA.memory_status()` — that gives the actual per-PINN footprint including optimizer state and AD tape. Then set `per_pinn_estimate` to that measured value + 20% headroom. (2) **Runtime probing**: start with batch_size=2, run one batch, call `CUDA.available_memory()` again, and auto-scale up if there's headroom. Option 1 is simpler and good enough for a first pass.

### Step 3: Reconstruct best weights from objective_matrix

After the parallel loop, derive `best_weights` from the matrix indices rather than tracking during iteration:

```julia
best_idx = argmin(objective_matrix)
j_best, i_best = Tuple(best_idx)
best_w1 = weight1_values[i_best]
best_w2 = weight2_values[j_best]
best_weights = create_weight_tuple(weight1, best_w1, weight2, best_w2, fixed_weights)
best_objective = objective_matrix[best_idx]


```

@architect: Wait... this is way better good job

@claude-opus-4.6: Yeah — the key win is eliminating shared mutable state entirely. No locks, no atomics, no race conditions. Just fill the matrix in parallel, then scan it once at the end. Simpler code AND correct by construction.

### Step 4: Add progress reporting

Since threads print concurrently (garbled output), replace per-config `println` with a thread-safe counter:

```julia
completed = Threads.Atomic{Int}(0)
total = length(all_configs)

# Inside each spawned task, after evaluate:
Threads.atomic_add!(completed, 1)
@info "Grid search progress" completed=completed[] total=total thread=Threads.threadid()
```

@architect: After running the implementation I had further edits. The progress bars were a mess with 20 threads fighting for the terminal. I wanted one shared bar per batch so you can see all PINNs training together. Also needed a startup banner showing threads/GPU info, and main.jl was a mess of commented blocks — needed proper toggles.

@claude-opus-4.6: Implemented all of the above. Here's what changed beyond the original plan:
- `train_pinn()` now takes `progress_callback::Union{Function,Nothing}=nothing` — grid search passes a shared `ProgressMeter.Progress` bar, single run creates its own bar as before.
- `evaluate_weight_configuration()` threads the callback through to `train_pinn()`.
- `grid_search_2d()` creates ONE shared progress bar per batch (total = `batch_size * iters_per_pinn`). All threads tick it via `ProgressMeter.next!()` (internally SpinLock-protected). Shows live `:loss` and `:thread`.
- Startup banner prints threads, GPU name, VRAM free, grid dims, batch size, weight ranges.
- `ProgressBar.jl` had thread-unsafe `global` counters — converted to closured `Ref`s inside `Bar()`.
- `main.jl` now uses `TRAINING_MODE = "GRID_SEARCH" | "SINGLE_RUN" | "SCALING_ADAM"` with `if/elseif` — no more commenting/uncommenting blocks.
- Fixed `PINNSettings` constructor call in `evaluate_weight_configuration` (15 args → 14, added type casts).

## What Does NOT Change

- `loss_fn()` / loss functions — untouched
- `gpu_utils.jl` — untouched (global Refs are read-only after init)
- `initialize_network()` — untouched (`Random.default_rng()` is task-local in Julia 1.9+)
- File I/O — each config writes to its own unique `config_dir`, no collisions

## Thread Safety Analysis

| Concern | Status | Why |
|---------|--------|-----|
| `objective_matrix[j,i]` writes | Safe | Each (i,j) is unique per task |
| `best_objective` tracking | Fixed | Post-hoc reduction, no shared mutation |
| File I/O (CSV, JSON) | Safe | Unique `config_dir` per weight config |
| `Random.default_rng()` | Safe | Task-local in Julia 1.9+ |
| GPU memory allocation (`CUDA.cu`) | Safe | CUDA.jl handles multi-threaded access, separate streams per thread |
| `GPUUtils.GPU_AVAILABLE` Ref | Safe | Read-only after `__init__()` |
| Zygote AD tape | Safe | Each gradient call builds its own tape |

## Usage Requirement 

Julia must be launched with threads enabled:
```bash
julia -t 8 src/main.jl        # 8 threads
julia -t auto src/main.jl     # auto-detect cores
```

@architect: talk to me how you know this is true

@claude-opus-4.6: Julia defaults to 1 thread. `Threads.@spawn` creates tasks, but tasks only run in parallel if there are multiple OS threads. The `-t N` flag (or `JULIA_NUM_THREADS=N` env var) sets thread count at startup — it cannot be changed after Julia starts. You can verify with `Threads.nthreads()` at the REPL. Without `-t N`, the spawned tasks will run cooperatively on a single thread (still correct, just no speedup). So this is a launch-time requirement, not a code change.
