# GPU Memory Pressure — Buffer Pre-allocation

**Status:** Open
**Priority:** High
**Created:** February 24, 2026

## Problem

Training crashes or slows severely on GPU due to repeated memory allocation/deallocation every loss evaluation. With 100k iterations and 100 training examples, this means ~10 million redundant CPU→GPU transfers per run.

### 1. Repeated GPU transfers every loss call (HOT PATH)

**File:** `modelcode/PINN.jl:L167-171`

Inside `loss_fn`, the `Zygote.ignore()` block calls `GPUUtils.to_device()` on `ode_matrix_flat`, `boundary_condition`, `data`, and `xs` **every single time** the loss function is called. These arrays are constants for a given ODE — they never change during training.

```julia
# Called EVERY loss evaluation — allocates new GPU memory each time
ode_flat_dev, bc_dev, data_dev, xs_dev = Zygote.ignore() do
    (GPUUtils.to_device(ode_matrix_flat; gpu=use_gpu),
     GPUUtils.to_device(boundary_condition; gpu=use_gpu),
     GPUUtils.to_device(data; gpu=use_gpu),
     GPUUtils.to_device(collect(settings.xs); gpu=use_gpu))
end
```

### 2. W matrix rebuilt every loss call

**File:** `utils/loss_functions.jl:L41-61`

`generate_loss_pde_value` rebuilds the entire W operator matrix inside `Zygote.ignore()` on every loss evaluation. This includes:
- `Float32.(collect(...))` allocations
- Triple-nested loop to populate `W_cpu`
- `similar()` + `copyto!()` to transfer to GPU

The W matrix depends only on `xs`, `ode_matrix_flat`, and `INV_FACT` — all constants during training.

### 3. Power vectors rebuilt every loss call

**File:** `utils/loss_functions.jl:L80-96`

`generate_loss_bc_value` recomputes `pow_u`, `pow_du_full`, and boundary conditions + transfers them to GPU on every evaluation.

### 4. Padded data + mask rebuilt every loss call

**File:** `utils/loss_functions.jl:L117-129`

`generate_loss_supervised_value` re-allocates `padded_data` and `mask` on every evaluation.

### 5. Sequential ODE iteration with per-item transfers

**File:** `modelcode/PINN.jl:L206-241`

`global_loss` loops over every ODE in the training set, calling `loss_fn` for each. Each call triggers its own set of GPU transfers (issue #1). With 100 training examples x 100k iterations = 10 million redundant transfer batches.

## Solution — Pre-allocated GPU Buffers

### Strategy

Transfer all constant data to GPU **once** before the optimization loop and reuse as pre-allocated buffers.

### Per-ODE buffers (computed once per ODE, before training)

| Buffer | Source | Current location |
|--------|--------|-----------------|
| `ode_matrix_flat_gpu` | `vec(alpha_matrix_key)` | `PINN.jl:L219` |
| `boundary_condition_gpu` | `[series_coeffs[1], series_coeffs[2]]` | `PINN.jl:L220` |
| `data_gpu` | `series_coeffs` | `PINN.jl:L221` |
| `W_matrix_gpu` | Composite operator from `xs`, `ode_flat`, `INV_FACT` | `loss_functions.jl:L41-61` |
| `pow_u_gpu` | Power vector for u(x0) | `loss_functions.jl:L82` |
| `pow_du_gpu` | Power vector for Du(x0) | `loss_functions.jl:L85` |
| `padded_data_gpu` | Zero-padded supervised data | `loss_functions.jl:L118-121` |
| `mask_gpu` | Supervised mask | `loss_functions.jl:L122` |

### Shared buffers (computed once, shared across all ODEs)

| Buffer | Source | Current location |
|--------|--------|-----------------|
| `xs_gpu` | `collect(settings.xs)` | `PINN.jl:L171` |

### Implementation Steps

1. **Create a `PrecomputedBuffers` struct** — holds all GPU-resident constant arrays for one ODE
2. **Create a `precompute_buffers` function** — takes a training dict + settings, returns `Dict{Matrix, PrecomputedBuffers}` with everything already on GPU
3. **Call `precompute_buffers` once** in `train_pinn` before the optimization loop
4. **Pass buffers into `global_loss` / `loss_fn`** — replace the `Zygote.ignore()` transfer blocks with direct buffer reads
5. **Update loss functions** to accept pre-computed W, power vectors, etc. as arguments instead of recomputing

### Future: Batched forward pass

Instead of looping over ODEs sequentially in `global_loss`, stack all ODE inputs into a single batched tensor and run one forward pass through the network. This is a larger refactor but would maximize GPU utilization.

## Files to Change

| File | Changes |
|------|---------|
| `modelcode/PINN.jl` | Add `PrecomputedBuffers` struct, `precompute_buffers()`, pass buffers through `train_pinn` → `global_loss` → `loss_fn` |
| `utils/loss_functions.jl` | Accept pre-computed W, power vectors, padded data as args instead of recomputing in `Zygote.ignore()` |
| `utils/training_schemes.jl` | Pass buffer precomputation through `scaling_adam` and other training modes |

## Stale Memory Note

`memory/snapshot-feature-flow.md:L8-11` still references `model-snapshots/` paths instead of `results/run-{id}/snapshots/`. Should be updated separately.
