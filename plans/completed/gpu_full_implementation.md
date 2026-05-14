# GPU Acceleration Full Implementation

**Status:** Implemented - Needs Testing
**Priority:** Medium
**Created:** February 5, 2026
**Updated:** February 5, 2026

## Problem (Resolved)

GPU implementation was disabled due to Zygote/Lux/CUDA.jl v5 compatibility issues:

1. `LossFunctionSettings` struct had concrete `Vector{Float32}` types — couldn't hold CuArrays
2. Loss functions used scalar-indexed loops (`a_vec[i] * x^(i-1)`) incompatible with GPU
3. Zygote's `∇getindex!` rule crashed on CuArrays during backpropagation
4. `use_gpu = false` was hardcoded

## Solution Applied

### 1. Parametric Struct (`utils/loss_functions.jl`)
- `LossFunctionSettings{V<:AbstractVector{Float32}}` — accepts both `Vector` and `CuVector`

### 2. Vectorized Loss Functions (`utils/loss_functions.jl`)
- **PDE loss**: Precomputes composite operator `W` (constant w.r.t. `a_vec`) inside `Zygote.ignore()`, then `residual = W * a_vec` — single differentiable matrix-vector multiply, no scalar indexing
- **BC loss**: Full-length dot products with precomputed power vectors instead of scalar loops. Padded derivative weights to avoid slicing `a_vec`
- **Supervised loss**: Padded data + mask arrays to avoid `a_vec[1:K]` slicing that triggers broken `∇getindex!`

### 3. Device Transfer (`utils/gpu_utils.jl`, `modelcode/PINN.jl`)
- Added `GPUUtils.to_device(x; gpu=false)` for clean CPU/GPU transfers
- `loss_fn` transfers all inputs to correct device before network forward pass
- `initialize_network` moves parameters to GPU via `CUDA.cu(ComponentArray)`
- `train_pinn` auto-detects GPU: `use_gpu = GPUUtils.is_gpu_available()`
- Trained parameters transferred back to CPU for evaluation/plotting

### 4. Other Fixes
- `Project.toml`: Fixed compat key `cuda` → `CUDA`
- Pre-computed `INV_FACT` (Float32 inverse factorials) replaces BigFloat `fact` in training path

## Known Limitations

- **LBFGS on GPU**: `OptimizationOptimJL.LBFGS()` may not fully support CuArrays. If it crashes, consider adding an Adam stage via `OptimizationOptimisers.Adam()` for GPU, then LBFGS on CPU for fine-tuning.
- **Small network**: With only ~100 neurons and 10 collocation points, GPU overhead may not yield speedup. Benefit increases with larger networks/more points.

## Files Changed

| File | Changes |
|------|---------|
| `utils/loss_functions.jl` | Parametric struct, vectorized GPU-native losses |
| `utils/gpu_utils.jl` | Added `to_device()` helper |
| `modelcode/PINN.jl` | GPU detection, device transfers, enabled GPU path |
| `Project.toml` | Fixed CUDA compat key |

## Test

```julia
julia --project=. src/main.jl
```

Should show (if GPU available):
```
[ Info: GPU detected: NVIDIA GeForce RTX 3060
[ Info: Training on GPU: NVIDIA GeForce RTX 3060
```

Or (if no GPU):
```
[ Warning: GPU not available, falling back to CPU
[ Info: Training on CPU
```
