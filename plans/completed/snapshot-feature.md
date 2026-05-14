# Milestone Snapshots & Inference Replay

**Date Created:** February 14, 2026
**Updated:** February 16, 2026
**Status:** Plan — ready for implementation
**Participants:** @architect, @claude-opus-4.6

---

## The Problem

Training runs can be long (1M iterations in `scaling_adam`). Right now:
- The milestone callback logs error values but **doesn't persist the model weights**.
- To inspect what the model "knew" at iteration 5000, you have to **retrain from scratch**.
- If a run crashes or is interrupted, all learned weights are lost.
- There's no way to initialize a new training run from a previously trained model.

@architect: Good understanding of the problem

---

## Design: Three Layers

All three layers share a single artifact — a raw `.bin` file of the `Float32` parameter vector (~88KB per snapshot). Each layer builds on the previous one. 

### Layer 1: Save weights at milestones

The existing `on_milestone` callback in `scaling_adam` (`training_schemes.jl:L93-L98`) already receives `p_current` and runs `evaluate_solution` to get coefficients. It stores iteration, loss, and coefficients in the `milestones` array — which later gets written to `scaling_adam_results.json` (`training_schemes.jl:L119-L155`).

**The only addition**: write `p_current` to a `.bin` file inside `on_milestone`.

```julia
# Inside on_milestone — one new line
snapshot_path = joinpath(snapshot_dir, "iter-$(lpad(iteration, 7, '0')).bin")
write(snapshot_path, Float32.(vec(p_current)))
```

The coefficients + losses at each milestone are **already saved** in the existing `scaling_adam_results.json` output. The `.bin` files just add the weights alongside.

@architect: Exactly how I want the implementation

**Directory structure:**

```
snapshots/
└── run-adam-a7x9k2m1/
    ├── iter-0001000.bin       # ~88KB each
    ├── iter-0005000.bin
    ├── iter-0010000.bin
    └── ...
```

@architect: ACTUALLY, make the dir (model-snapshots)

The matching metadata (iteration, loss, coefficients) lives in `results/scaling_adam_results.json` which already indexes milestones. No new manifest file needed. @architect: Good

### Layer 2: Load weights for inference

Given a `.bin` snapshot and a `PINNSettings`, reconstruct the network and run a forward pass.

```julia
function load_and_infer(snapshot_path::String, settings::PINNSettings, ode_matrix::Matrix)
    # 1. Rebuild network from settings (deterministic from neuron_num, n_terms, seed)
    coeff_net, _, st = initialize_network(settings)

    # 2. Load saved weights into the same ComponentArray layout
    raw = reinterpret(Float32, read(snapshot_path))
    p_template = ComponentArray(Lux.setup(Random.default_rng(), coeff_net)[1])
    p = ComponentArray(raw, getaxes(p_template))

    # 3. Run inference
    matrix_flat = Float32.(vec(ode_matrix))
    coefficients = first(coeff_net(matrix_flat, p, st))[:, 1]

    return coefficients
end
```

This works because `initialize_network` builds the same architecture deterministically from `PINNSettings` (`PINN.jl:L128-L159`), so the `ComponentArray` axis layout is always identical. The `.bin` file just supplies the learned values.

**Replay across snapshots**: sweep every `.bin` in a run directory with a fixed ODE matrix to see how the model's coefficient predictions evolve over training. @architect: Something like this would work better for the nn-viewer that we have. But we can talk about this later.

### Layer 3: Load weights to resume training

Same `.bin` file, but instead of running inference, use it to **initialize a new training run**. The change is in `train_pinn` (`PINN.jl:L260`):

```julia
function train_pinn(settings::PINNSettings, output_dir;
                    snapshot_path::Union{String,Nothing}=nothing,  # NEW kwarg
                    milestone_interval::Int=0,
                    on_milestone::Union{Function,Nothing}=nothing,
                    ...)

    coeff_net, p_init_ca, st = initialize_network(settings; use_gpu=use_gpu)

    # If snapshot provided, override random init with saved weights
    if snapshot_path !== nothing
        raw = reinterpret(Float32, read(snapshot_path))
        p_init_ca = ComponentArray(raw, getaxes(p_init_ca))
        if use_gpu
            p_init_ca = CUDA.cu(p_init_ca)
        end
        @info "Loaded weights from snapshot: $snapshot_path"
    end

    # ... rest of train_pinn unchanged — p_init_ca goes into OptimizationProblem
    prob = OptimizationProblem(optfun, p_init_ca)
```

The Adam optimizer starts fresh (no momentum history from the previous run). This is acceptable — Adam reconverges quickly and the weights themselves carry the learned knowledge.

@architect: This is good.

---

## What Gets Saved vs. What Already Exists

| Data | Where | New? |
|------|-------|------|
| Model weights at each milestone | `snapshots/run-{id}/iter-NNNNNNN.bin` | **NEW** |
| Coefficients + losses at each milestone | `results/scaling_adam_results.json` | Already exists (`training_schemes.jl:L142-L149`) |
| Loss history (every iteration) | `results/loss.csv` | Already exists (`PINN.jl:L348-L357`) |
| PINNSettings / hyperparameters | `results/scaling_adam_results.json` metadata | Already exists (`training_schemes.jl:L121-L136`) |

The only new artifact is the `.bin` weight files.

---

## What We Do NOT Save

- **Optimizer state** (Adam momentum `m̂`, `v̂`) — not accessible from `Optimization.jl` callbacks, and not needed for inference. Acceptable loss for Layer 3 resume.
- **Lux model struct** — reconstructible from `PINNSettings` via `initialize_network()`.
- **Lux state `st`** — empty/static for our architecture (Dense layers only, no BatchNorm/Dropout).
- **Full training history per snapshot** — already in `loss.csv`.

@architect: Good good

---

## Files That Change

| File | Change | Complexity |
|------|--------|------------|
| `utils/training_schemes.jl` | Add `write()` call inside `on_milestone`, create snapshot dir at training start | Low — ~5 lines |
| `modelcode/PINN.jl` | Add `snapshot_path` kwarg to `train_pinn`, conditional weight loading | Low — ~10 lines |
| NEW: `utils/snapshot_utils.jl` | `load_and_infer()`, `replay_snapshots()` helper functions | Medium — new file, ~60 lines |

---

## Implementation Order

1. **Layer 1** — Add `write()` to `on_milestone` in `scaling_adam`. Create `snapshots/run-{id}/` dir.
2. **Layer 2** — Create `snapshot_utils.jl` with `load_and_infer()` and `replay_snapshots()`.
3. **Layer 3** — Add `snapshot_path` kwarg to `train_pinn` for warm-start training.

Each layer is independently useful and can be shipped/tested separately.

---

## Future: TF SavedModel Export (Scale-Up)

Lux.jl provides `Lux.Serialization.export_as_tf_saved_model` which exports to TensorFlow's universal SavedModel format. This would allow:
- Running inference outside Julia (Python, JS, web deployment)
- Handing trained models to collaborators without Julia
- Serving models via TF Serving

Requires `Reactant.jl` + `PythonCall.jl` — heavy dependencies. Worth revisiting if we scale up the model or need cross-language interop. Not in scope for v1.

Reference: https://lux.csail.mit.edu/stable/api/Lux/serialization

---

## References

- Existing milestone callback: `utils/training_schemes.jl:L93-L98`
- Existing milestone JSON output: `utils/training_schemes.jl:L119-L155`
- Network initialization: `modelcode/PINN.jl:L128-L159`
- train_pinn signature: `modelcode/PINN.jl:L260`
- OptimizationProblem creation: `modelcode/PINN.jl:L309`
- Inference pattern: `modelcode/PINN.jl:L385`
- Lux.jl serialization API: https://lux.csail.mit.edu/stable/api/Lux/serialization
- Lux.jl TrainState (not used, but noted): https://lux.csail.mit.edu/stable/api/Lux/utilities
