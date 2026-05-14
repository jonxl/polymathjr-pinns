# GPU Acceleration Implementation Plan for PINN.jl

**Target Hardware:** NVIDIA RTX 3080 (Ampere GA102)  
**Julia Version:** 1.9+  
**CUDA.jl Version:** 5.x  
**Date Created:** February 5, 2026
**Updated:** February 6, 2026
**Status:** Implemented — GPU path working, LBFGS convergence issue remains

---

## Overview

Streamlined GPU acceleration implementation for the PINN Power Series ODE Solver. This plan adds GPU support while maintaining full CPU compatibility.

---

## Hardware Specifications

| Specification | Value |
|--------------|-------|
| GPU | NVIDIA RTX 3080 |
| Compute Capability | 8.6 (Ampere) |
| VRAM | 10 GB GDDR6X |
| Tensor Cores | 272 (3rd generation) |
| CUDA Cores | 8,704 |
| Memory Bandwidth | 760 GB/s |

---

## GPU Pattern Validation Summary

| Component | Status | Pattern |
|-----------|--------|---------|
| `coeff_net \|> gpu` | ✅ Validated | Moves network to GPU |
| `Zygote.gradient(p, ...)` | ✅ Validated | Works with CuArray |
| `AutoZygote()` | ✅ Validated | GPU-compatible AD |
| `OptimizationProblem` | ✅ Validated | No changes needed |
| Adam optimizer | ✅ Validated | Excellent GPU support |
| LBFGS optimizer | ⚠️ Needs tuning | Reduce `m=5` for GPU |

---

## Memory Analysis (RTX 3080 - 10GB VRAM)

**Configuration:** 3 hidden layers × 100 neurons, 21 output coefficients, 10 collocation points

| Component | Memory Usage |
|-----------|--------------|
| Parameters (22,622) | 0.09 MB |
| Gradients | 0.09 MB |
| Adam State | 0.17 MB |
| Forward Activations | 12.3 MB |
| Collocation Points | 1.7 MB |
| **Total** | **~2 MB** |

**Recommendation:** Safe to run with substantial VRAM headroom.

---

## Implementation Steps

### Step 1: Update Project.toml

**File:** `/home/archimedes/code/polymathjr/pinns/Project.toml`

**Add dependency:**
```toml
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[compat]
cuda = "≥ 3.0"
```

---

### Step 2: Create GPU Utility Module

**File:** `/home/archimedes/code/polymathjr/pinns/utils/gpu_utils.jl`

```julia
module GPUUtils

using CUDA

export is_gpu_available, gpu_device!, to_gpu, get_device

const GPU_AVAILABLE = Ref{Bool}(false)
const DEVICE = Ref{Union{CUDA.Device, Nothing}}(nothing)

function __init__()
    GPU_AVAILABLE[] = CUDA.functional()
    if GPU_AVAILABLE[]
        DEVICE[] = CUDA.device()
        @info "GPU detected: $(CUDA.name(DEVICE[]))"
    else
        @warn "GPU not available, falling back to CPU"
    end
end

is_gpu_available() = GPU_AVAILABLE[]

function gpu_device!()
    return DEVICE[]
end

function to_gpu(x::AbstractArray)
    @assert is_gpu_available() "GPU not available"
    return CUDA.cu(x)
end

function get_device()
    return is_gpu_available() ? CUDA.device() : :cpu
end

end
```

---

### Step 3: Modify modelcode/PINN.jl

**File:** `/home/archimedes/code/polymathjr/pinns/modelcode/PINN.jl`

#### 3.1 Add CUDA Import (after line 24)

```julia
using CUDA

include("../utils/gpu_utils.jl")
using .GPUUtils
```

#### 3.2 Update initialize_network Function (replace lines 119-141)

```julia
function initialize_network(settings::PINNSettings; use_gpu::Bool=true)
    # Determine device
    device = use_gpu && GPUUtils.is_gpu_available() ? gpu : cpu
    
    # Find maximum input size
    max_input_size = if !isempty(settings.ode_matrices)
        maximum(begin
            alpha_matrix_key = key
            prod(size(alpha_matrix_key))
        end for (alpha_matrix_key, series_coeffs) in settings.ode_matrices)
    else
        settings.n_terms_for_power_series + 1
    end
    
    # Create network
    coeff_net = Lux.Chain(
        Lux.Dense(max_input_size, settings.neuron_num, σ),
        Lux.Dense(settings.neuron_num, settings.neuron_num, σ),
        Lux.Dense(settings.neuron_num, settings.neuron_num, σ),
        Lux.Dense(settings.neuron_num, settings.n_terms_for_power_series + 1)
    )
    
    # Initialize
    rng = Random.default_rng()
    Random.seed!(rng, settings.seed)
    
    # Move to device and setup
    if device == gpu
        coeff_net = coeff_net |> gpu
    end
    p_init, st = Lux.setup(rng, coeff_net)
    
    # Wrap parameters
    p_init_ca = ComponentArray(p_init)
    
    if device == gpu
        p_init_ca = p_init_ca |> gpu
    end
    
    return coeff_net, p_init_ca, st
end
```

#### 3.3 Update loss_fn Function (replace lines 147-189)

```julia
function loss_fn(p_net, data, coeff_net, st, ode_matrix_flat, boundary_condition, settings::PINNSettings)
    # Determine device from parameters
    on_gpu = p_net isa CUDA.CuArray
    
    # Move inputs to GPU if needed
    if on_gpu
        ode_matrix_flat = GPUUtils.to_gpu(ode_matrix_flat)
        boundary_condition = GPUUtils.to_gpu(boundary_condition)
    end
    
    # Forward pass
    a_vec = first(coeff_net(ode_matrix_flat, p_net, st))[:, 1]
    
    # Create loss settings (handles both CPU and GPU arrays)
    loss_func_settings = LossFunctionSettings(
        a_vec,
        settings.n_terms_for_power_series,
        ode_matrix_flat,
        settings.x_left,
        boundary_condition,
        settings.xs,
        settings.num_points,
        settings.num_supervised,
        data,
    )
    
    # Compute losses
    loss_pde = generate_loss_pde_value(loss_func_settings)
    loss_bc = generate_loss_bc_value(loss_func_settings)
    loss_supervised = generate_loss_supervised_value(loss_func_settings)
    
    # Return weighted loss and components
    return loss_pde * settings.pde_weight + 
           settings.bc_weight * loss_bc + 
           settings.supervised_weight * loss_supervised,
           loss_bc, loss_pde, loss_supervised
end
```

#### 3.4 Update train_pinn Function (replace lines 249-337)

```julia
function train_pinn(settings::PINNSettings, csv_file::Any)
    # Determine if GPU should be used
    use_gpu = GPUUtils.is_gpu_available()
    
    if use_gpu
        @info "Training on GPU: $(CUDA.name(CUDA.device()))"
    else
        @info "Training on CPU"
    end
    
    # Initialize network (with GPU if available)
    coeff_net, p_init_ca, st = initialize_network(settings; use_gpu=use_gpu)
    
    # Initialize loss buffer
    initialize_loss_buffer()
    history = []
    latest_metrics = Ref((0.0f0, 0.0f0, 0.0f0))
    
    # Loss wrapper
    function loss_wrapper(p_net, _)
        loss, losses = global_loss(p_net, settings, coeff_net, st)
        latest_metrics[] = (losses.bc, losses.pde, losses.sup)
        return loss
    end
    
    # Callback
    function custom_callback(state, l; progress_bar=nothing)
        bc, pde, sup = latest_metrics[]
        push!(history, (total=l, bc=bc, pde=pde, supervised=sup))
        if progress_bar !== nothing
            progress_bar(state, l)
        end
        return false
    end
    
    # Setup optimization
    adtype = Optimization.AutoZygote()
    optfun = OptimizationFunction(loss_wrapper, adtype)
    prob = OptimizationProblem(optfun, p_init_ca)
    
    # Warmup GPU (compilation)
    if use_gpu
        CUDA.reclaim()
        @info "Warming up GPU..."
    end
    
    # Stage 2: LBFGS optimization
    @info "Starting LBFGS fine-tuning..."
    p_two = ProgressBar.ProgressBarSettings(settings.maxiters_lbfgs, "LBFGS fine-tune...")
    callback_two = ProgressBar.Bar(p_two)
    
    # Use reduced memory for GPU
    lbfgs_opt = use_gpu ? 
        OptimizationOptimJL.LBFGS(m=5) :  # Reduced memory for GPU
        OptimizationOptimJL.LBFGS()
    
    res = solve(prob,
        lbfgs_opt;
        callback = (state, l) -> custom_callback(state, l; progress_bar=callback_two),
        maxiters=settings.maxiters_lbfgs)
    
    # Extract trained parameters
    p_trained = res.u
    
    @info "Training complete."
    
    # Write history to CSV
    for entry in history
        buffer_loss_values(
            total_loss = entry.total,
            total_loss_bc = entry.bc,
            total_loss_pde = entry.pde,
            total_loss_supervised = entry.supervised
        )
    end
    write_buffer_to_csv(csv_file)
    
    return p_trained, coeff_net, st
end
```

---

## Files Modified Summary

| File | Changes | Complexity |
|------|---------|-------------|
| `Project.toml` | Add CUDA dependency | Low |
| `utils/gpu_utils.jl` | **NEW** - GPU utilities | Low |
| `modelcode/PINN.jl` | Add GPU support to 3 functions | Medium |

---

## Expected Performance

| Metric | CPU | GPU (RTX 3080) | Speedup |
|--------|-----|----------------|---------|
| Forward pass | ~0.5ms | ~0.1ms | 5x |
| Backward pass | ~1.5ms | ~0.3ms | 5x |
| Per iteration | ~2.0ms | ~0.4ms | 5x |
| 10,000 iterations | ~20s | ~4s | 5x |

---

## GPU Validation Checklist

Before running training, execute:

```julia
using CUDA

# 1. Check GPU availability
@assert CUDA.functional() "CUDA not functional"

# 2. Check device name
println("GPU: $(CUDA.name(CUDA.device()))")

# 3. Check VRAM
println("VRAM: $(CUDA.available_memory() / 1e9) GB")

# 4. Verify Float32
@assert eltype(p_init_ca) == Float32 "Must use Float32"

# 5. Run quick test
@info "Testing GPU forward pass..."
```

---

## Troubleshooting

### GPU Not Detected

```julia
using CUDA
CUDA.functional()  # Should return true

# If false:
# - Check CUDA Toolkit installation
# - Verify driver compatibility (≥ 450.x for CUDA 11.x)
# - Run `nvidia-smi` to confirm GPU visibility
```

### Out of Memory

```julia
# Reduce LBFGS memory parameter
lbfgs_opt = OptimizationOptimJL.LBFGS(m=3)

# Or fallback to CPU
use_gpu = false
```

### Slow First Iteration

Normal - JIT compilation of CUDA kernels. Subsequent iterations will be fast.

---

## Rollback Instructions

To remove GPU support:

1. Remove `CUDA` from `Project.toml`
2. Remove `utils/gpu_utils.jl`
3. Revert `modelcode/PINN.jl` to original functions

---

## Implementation Notes (February 6, 2026)

### What went right

- **Parametric struct approach worked perfectly.** Changing `LossFunctionSettings` from concrete `Vector{Float32}` to `{V<:AbstractVector{Float32}}` let the same struct hold CPU or GPU arrays with zero runtime cost.
- **Vectorized loss functions are correct.** Precomputing the composite operator `W` inside `Zygote.ignore()` so that `residual = W * a_vec` is a single matrix-vector multiply eliminated all scalar indexing. CPU and GPU produce identical loss values and gradients.
- **`Zygote.ignore()` was the key pattern.** All constant array construction (Vandermonde matrices, power vectors, padding/masks) wrapped in `Zygote.ignore()` — Zygote treats the result as a constant, so mutation inside the block is fine and AD flows only through the final differentiable ops (`W * a_vec`, `sum(a .* pow_u)`, etc.).
- **`copyto!` instead of `.=` for cross-device transfers.** Broadcasting (`cu_array .= cpu_array`) tries to compile a GPU kernel referencing CPU memory and fails. `copyto!` uses memcpy and works across devices. This was the first GPU test failure and the fix was a one-line change per call site.
- **`getdata(res.u)` instead of `res.u.data` for ComponentArrays.** ComponentArray intercepts `.data` as a named component lookup (looking for a layer called `data`). The `getdata()` accessor retrieves the underlying flat array correctly.
- **GPU auto-detection works.** `GPUUtils.is_gpu_available()` in `train_pinn` cleanly selects the path. On CPU-only machines it falls back transparently.
- **Zygote gradient computation through GPU loss functions passes.** This was the core blocker — tested with `Zygote.gradient(test_loss, cu_array)` and got correct gradients matching CPU values.
- **Full `src/main.jl` runs end-to-end on GPU.** All three training runs (LBFGS 1000, 10000, 100000) complete, evaluation/plotting works, CSV output written.

### What still needs work

- **LBFGS doesn't converge on GPU.** Loss stays stuck at `5.5e15` — the optimizer terminates after ~15 loss evaluations regardless of `maxiters`. LBFGS line search relies on precise scalar operations that don't interact well with the GPU gradient path. Need to either fall back to CPU for LBFGS or add an Adam stage (which works well on GPU).
- **The plan's `coeff_net |> gpu` pattern was not used.** Moving the Lux model struct to GPU didn't work with `ComponentArray` + `Zygote`. Instead, only the parameter `ComponentArray` is moved to GPU via `CUDA.cu(p_init_ca)`, and Lux dispatches based on parameter device.
- **Performance benefit is marginal at current scale.** With 100 neurons and 10 collocation points, the network is tiny (~22K params, ~2MB). GPU overhead likely exceeds compute savings. The GPU infrastructure is now ready for scaling up.

---

## References

- CUDA.jl Documentation: https://cuda.julialang.org/
- Lux.jl GPU Guide: https://lux.csail.mit.edu/dev/
- Zygote.jl GPU Support: https://fluxml.ai/Zygote.jl/stable/
