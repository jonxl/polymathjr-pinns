# Per-Layer Named Tensor Safetensors

**Status:** Open
**Priority:** High
**Created:** June 3, 2026
**Revised:** June 3, 2026

## Context

@architect: The current safetensors format saves all model weights as a single flat `"weights"` blob. To load the model you also need the `PINNSettings` (neuron count, layers, activation) passed separately — the file is not self-contained. The file should be structured like a proper HuggingFace model: per-layer named tensors (`layer_1.weight`, `layer_1.bias`, etc.) plus embedded architecture metadata, so the file IS the model.

Additionally, the legacy `.bin` raw-byte format and v1 flat safetensors format are being **completely removed**. There is only one format going forward: v2 per-layer safetensors.

## Current State

**`utils/safetensors_utils.jl`** (101 lines):
- `save_safetensors_vector` — writes single flat `"weights"` tensor -> **will be replaced**
- `load_safetensors_vector` — reads single flat tensor -> **will be replaced**
- `load_snapshot_vector` — dispatches `.safetensors` vs `.bin` -> **will be removed**
- `_write_u64_le`, `_read_u64_le` — binary helpers -> **kept**

**Integration points that write safetensors:**
- `training_schemes.jl:86-87` — checkpoint: `save_safetensors_vector(snapshot_path, p_current)`
- `training_schemes.jl:105-106` — final model: `save_safetensors_vector(final_model_path, p_trained)`

**Integration points that read safetensors:**
- `PINN.jl:317-327` — `initialize_network(settings)` always runs first for fresh weights, then if `snapshot_path` is provided, `load_snapshot_vector` overwrites the params. Architecture always comes from `settings`, not the file.
- `snapshot_utils.jl:24-37` — `load_and_infer(snapshot_path, settings, ode_matrix)` requires external `settings` to rebuild architecture
- `snapshot_utils.jl:49-67` — `replay_snapshots(run_dir, settings, ode_matrix)` passes settings through

**Network architecture** (`PINN.jl:139-144`):

```julia
Lux.Chain(
    Dense(max_input_size, neuron_num, sigma),   # layer_1: weight (N x in), bias (N)
    Dense(neuron_num, neuron_num, sigma),        # layer_2: weight (N x N), bias (N)
    Dense(neuron_num, neuron_num, sigma),        # layer_3: weight (N x N), bias (N)
    Dense(neuron_num, n_terms+1)                 # layer_4: weight (out x N), bias (out)
)
```

`Lux.setup` + `ComponentArray(p_init)` produces hierarchical leaf tensors accessible as `p_ca.layer_1.weight`, `p_ca.layer_1.bias`, `layer_2.weight`, etc. in deterministic property-name order.

## Target Format (v2 — the only format)

```json
{
  "layer_1.weight": {"dtype": "F32", "shape": [100, 3], "data_offsets": [0, 1200]},
  "layer_1.bias":   {"dtype": "F32", "shape": [100],   "data_offsets": [1200, 1600]},
  "layer_2.weight": {"dtype": "F32", "shape": [100, 100], "data_offsets": [1600, 41600]},
  "layer_2.bias":   {"dtype": "F32", "shape": [100],   "data_offsets": [41600, 42000]},
  "layer_3.weight": {"dtype": "F32", "shape": [100, 100], "data_offsets": [42000, 82000]},
  "layer_3.bias":   {"dtype": "F32", "shape": [100],   "data_offsets": [82000, 82400]},
  "layer_4.weight": {"dtype": "F32", "shape": [11, 100], "data_offsets": [82400, 86800]},
  "layer_4.bias":   {"dtype": "F32", "shape": [11],    "data_offsets": [86800, 86844]},
  "__metadata__": {
    "format": "polymathjr-pinns-model",
    "version": 2,
    "seed": 1234,
    "architecture": [
      {"type": "Dense", "in": 3, "out": 100, "activation": "sigmoid"},
      {"type": "Dense", "in": 100, "out": 100, "activation": "sigmoid"},
      {"type": "Dense", "in": 100, "out": 100, "activation": "sigmoid"},
      {"type": "Dense", "in": 100, "out": 11, "activation": "linear"}
    ]
  }
}
```

Each tensor preserves original shape (not flattened). Metadata fully describes the architecture. The file alone reconstructs the model.

## New Public API

Only three functions in `SafeTensorSnapshots`:

```julia
# Write — saves model weights with per-layer tensors + embedded architecture
save_safetensors_model(path, p_ca, coeff_net, seed; extra_metadata=Dict())

# Read low-level — returns raw metadata + tensor dict (for inspection/partial loading)
load_safetensors_model(path) -> (metadata::Dict, tensors::Dict{String, Array{Float32}})

# Read high-level — fully self-contained model reconstruction from just a file path
load_model(path) -> (coeff_net, p_ca, st, metadata::Dict)
```

**`load_model` is the key function.** Given just a file path, it reads metadata, rebuilds the `Lux.Chain` from the architecture spec, calls `Lux.setup` with the saved seed, fills in all weights from named tensors, and returns a ready-to-use model. **No external `PINNSettings` needed.** The file IS the model.

Internal reconstruction: read all named tensors -> rebuild `Lux.Chain` from metadata -> `Lux.setup` with saved seed to get template `ComponentArray` -> traverse template for flattening order -> assemble flat `Float32` vector -> `ComponentArray(flat, getaxes(template))`. This reuses the proven ComponentArray reconstruction path.

### Role of `initialize_network` after this change

`initialize_network` still exists and is used for **fresh training starts** (when no `snapshot_path` is provided). It reads architecture dimensions from `settings` and initializes random weights. When warm-starting, `load_model` replaces it entirely — architecture and weights come from the file, not from `settings`.

In `train_pinn`:

```julia
if snapshot_path !== nothing
    # Architecture + weights from the snapshot
    coeff_net, p_ca, st, _ = SafeTensorSnapshots.load_model(snapshot_path)
    p_init_ca = use_gpu ? CUDA.cu(p_ca) : p_ca
else
    # Fresh start — architecture from settings, random weights
    coeff_net, p_init_ca, st = initialize_network(settings; use_gpu=use_gpu)
end
```

`settings` still provides training configuration (loss weights, maxiters, ode_matrices, num_supervised, collocation points, domain bounds) regardless of which path is taken.

---

## Implementation Steps

### Step 1: Rewrite `safetensors_utils.jl`

**File:** `utils/safetensors_utils.jl`

**Remove:**
- `save_safetensors_vector`
- `load_safetensors_vector`
- `load_snapshot_vector`
- `DEFAULT_TENSOR_NAME` constant

**Keep:**
- `_write_u64_le` / `_read_u64_le` binary helpers

**Add private helpers:**

```julia
# Recursively collect leaf tensor names in deterministic ComponentArray order.
# Property names are sorted for deterministic roundtrips.
function _collect_leaf_names(x, prefix="")
    names = String[]
    if x isa ComponentArray
        for name in sort(string.(propertynames(x)))
            full_name = isempty(prefix) ? name : "$prefix.$name"
            child = getproperty(x, Symbol(name))
            append!(names, _collect_leaf_names(child, full_name))
        end
    elseif x isa AbstractArray
        push!(names, prefix)
    end
    return names
end

# Get a leaf tensor by dotted name (e.g., "layer_1.weight")
function _get_leaf(x, name::String)
    parts = split(name, ".")
    result = x
    for part in parts
        result = getproperty(result, Symbol(part))
    end
    return result
end

# Extract architecture metadata from a Lux.Chain
function _extract_architecture(chain)
    layers = []
    for layer in chain.layers
        act = layer.activation
        act_name = if act === Lux.sigma || string(act) == "sigmoid"
            "sigmoid"
        elseif act === identity
            "linear"
        elseif act === Lux.relu
            "relu"
        else
            string(act)
        end
        push!(layers, Dict(
            "type" => "Dense",
            "in" => size(layer.weight, 2),
            "out" => size(layer.weight, 1),
            "activation" => act_name
        ))
    end
    return layers
end

# Rebuild a Lux.Chain from architecture metadata
function _build_chain(architecture::Vector)
    layers = []
    for spec in architecture
        spec["type"] == "Dense" || error("Unknown layer type: $(spec["type"])")
        act = spec["activation"]
        activation_fn = if act == "sigmoid"
            Lux.sigma
        elseif act == "linear" || act == "none"
            identity
        elseif act == "relu"
            Lux.relu
        else
            error("Unknown activation: $act")
        end
        push!(layers, Lux.Dense(spec["in"], spec["out"], activation_fn))
    end
    return Lux.Chain(layers...)
end
```

**New public functions:**

```julia
"""
    save_safetensors_model(path, p_ca, coeff_net, seed; extra_metadata=Dict())

Write model weights as per-layer named tensors in Hugging Face safetensors format.
Architecture is extracted from `coeff_net` (a Lux.Chain) and embedded in `__metadata__`.
`Array()` calls on each leaf tensor transparently handle GPU -> CPU transfer.
"""
function save_safetensors_model(path::AbstractString, p_ca, coeff_net, seed::Int;
                                extra_metadata::Dict=Dict{String,Any}())
    architecture = _extract_architecture(coeff_net)

    metadata = Dict{String,Any}(
        "format" => "polymathjr-pinns-model",
        "version" => 2,
        "seed" => seed,
        "architecture" => architecture,
    )
    for (k, v) in extra_metadata
        metadata[k] = v
    end

    header = Dict{String,Any}()
    data_parts = Vector{UInt8}[]
    current_offset = 0

    for name in _collect_leaf_names(p_ca)
        arr = _get_leaf(p_ca, name)
        f32_arr = Float32.(Array(arr))  # Array() handles GPU->CPU
        raw_bytes = reinterpret(UInt8, vec(f32_arr))
        byte_len = length(raw_bytes)
        header[name] = Dict(
            "dtype" => "F32",
            "shape" => collect(size(f32_arr)),
            "data_offsets" => [current_offset, current_offset + byte_len],
        )
        push!(data_parts, raw_bytes)
        current_offset += byte_len
    end

    header["__metadata__"] = metadata

    header_bytes = Vector{UInt8}(codeunits(JSON.json(header)))
    mkpath(dirname(path))

    open(path, "w") do io
        _write_u64_le(io, UInt64(length(header_bytes)))
        write(io, header_bytes)
        for part in data_parts
            write(io, part)
        end
    end

    return path
end

"""
    load_safetensors_model(path) -> (metadata::Dict, tensors::Dict)

Low-level read of a per-layer safetensors file. Returns metadata and a dict
of named tensors (keyed by layer name, values are Float32 arrays in original shape).
"""
function load_safetensors_model(path::AbstractString)
    bytes = read(path)
    header_len = Int(_read_u64_le(bytes))
    8 + header_len <= length(bytes) || error("Invalid safetensors file: header exceeds file size")

    header = JSON.parse(String(bytes[9:8+header_len]))
    metadata = get(header, "__metadata__", Dict{String,Any}())

    tensors = Dict{String, Array{Float32}}()
    for (name, info) in header
        name == "__metadata__" && continue
        info["dtype"] == "F32" || error("Unsupported tensor dtype \"$(info["dtype"])\" for \"$name\". Expected F32.")
        shape = Tuple(info["shape"])
        offsets = info["data_offsets"]
        length(offsets) == 2 || error("Invalid data_offsets for \"$name\"")

        data_start = 8 + header_len + 1 + offsets[1]
        data_end   = 8 + header_len + offsets[2]
        data_start <= data_end <= length(bytes) || error("Tensor \"$name\" data exceeds file size")

        arr = reshape(reinterpret(Float32, bytes[data_start:data_end]), shape...)
        tensors[name] = arr
    end

    return metadata, tensors
end

"""
    load_model(path) -> (coeff_net, p_ca, st, metadata)

Fully self-contained model reconstruction from a safetensors file.
Rebuilds the network architecture from embedded metadata, initializes it
with the saved seed, and loads all per-layer weights.
Returns a ready-to-use model. No external PINNSettings needed.
"""
function load_model(path::AbstractString)
    metadata, tensors = load_safetensors_model(path)

    architecture = metadata["architecture"]
    seed = metadata["seed"]

    coeff_net = _build_chain(architecture)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    p_template, st = Lux.setup(rng, coeff_net)
    p_template_ca = ComponentArray(p_template)

    flat_parts = Float32[]
    for name in _collect_leaf_names(p_template_ca)
        haskey(tensors, name) || error("Missing tensor \"$name\" in safetensors file")
        append!(flat_parts, vec(Float32.(tensors[name])))
    end

    p_ca = ComponentArray(flat_parts, getaxes(p_template_ca))
    return coeff_net, p_ca, st, metadata
end
```

**Exports (replace old export line):**

```julia
export save_safetensors_model, load_safetensors_model, load_model
```

### Step 2: Simplify warm-start in `PINN.jl`

**File:** `architectures/PINN.jl`

The current code at lines 317-327 always runs `initialize_network` even when warm-starting, then overwrites params:

```julia
# Current (lines 317-327):
coeff_net, p_init_ca, st = initialize_network(settings; use_gpu=use_gpu)

if snapshot_path !== nothing
    raw = load_snapshot_vector(snapshot_path)
    p_init_ca = ComponentArray(raw, getaxes(p_init_ca))
    if use_gpu
        p_init_ca = CUDA.cu(p_init_ca)
    end
    @info "Loaded weights from snapshot: $snapshot_path"
end
```

**Replace with:**

```julia
if snapshot_path !== nothing
    coeff_net, p_ca, st, _ = SafeTensorSnapshots.load_model(snapshot_path)
    p_init_ca = use_gpu ? CUDA.cu(p_ca) : p_ca
    @info "Loaded model from snapshot: $snapshot_path"
else
    coeff_net, p_init_ca, st = initialize_network(settings; use_gpu=use_gpu)
end
```

`initialize_network` is skipped entirely when warm-starting. Architecture and weights both come from the file.

### Step 3: Update checkpoint saves in `training_schemes.jl`

**File:** `utils/training_schemes.jl`

Two single-line changes. Replace `save_safetensors_vector` with `save_safetensors_model`, passing `coeff_net` and `seed`:

**Checkpoint callback (line 87):**

```julia
# Before:
save_safetensors_vector(snapshot_path, p_current)

# After:
save_safetensors_model(snapshot_path, p_current, coeff_net, seed)
```

**Final model save (line 106):**

```julia
# Before:
save_safetensors_vector(final_model_path, p_trained)

# After:
save_safetensors_model(final_model_path, p_trained, coeff_net, seed)
```

Note: `coeff_net` is returned by `train_pinn` (line 98) and in scope. `seed` is a parameter of `run_training` (L59: `seed::Int=1234`). Both are available where needed.

### Step 4: Simplify `snapshot_utils.jl`

**File:** `utils/snapshot_utils.jl`

Remove `PINNSettings` dependency entirely. Both `load_and_infer` and `replay_snapshots` use `load_model` which is self-contained — no external settings, no `.bin` fallback, no version branching.

**Replace entire file with:**

```julia
module SnapshotUtils

using ComponentArrays
using Lux
import Random

include("../utils/safetensors_utils.jl")
using .SafeTensorSnapshots

"""
    load_and_infer(snapshot_path::String, ode_matrix::Matrix)

Load model from a `.safetensors` file and run inference to get coefficients.
The file is fully self-contained — no external PINNSettings needed.

Returns a vector of predicted coefficients.
"""
function load_and_infer(snapshot_path::String, ode_matrix::Matrix)
    coeff_net, p, st, _ = SafeTensorSnapshots.load_model(snapshot_path)
    matrix_flat = Float32.(vec(ode_matrix))
    coefficients = first(coeff_net(matrix_flat, p, st))[:, 1]
    return coefficients
end

"""
    replay_snapshots(run_dir::String, ode_matrix::Matrix)

Sweep every `.safetensors` snapshot in `run_dir`, run inference on each, and
return how coefficient predictions evolve over training.
No external settings needed — each file is self-contained.

Returns a vector of named tuples: [(iteration=Int, coefficients=Vector{Float32}), ...]
"""
function replay_snapshots(run_dir::String, ode_matrix::Matrix)
    snapshot_files = filter(f -> endswith(f, ".safetensors"), readdir(run_dir))
    sort!(snapshot_files)

    results = NamedTuple[]
    for fname in snapshot_files
        m = match(r"iter-(\d+)\.safetensors", fname)
        m === nothing && continue
        iteration = parse(Int, m.captures[1])

        snapshot_path = joinpath(run_dir, fname)
        coefficients = load_and_infer(snapshot_path, ode_matrix)
        push!(results, (iteration=iteration, coefficients=Vector{Float32}(coefficients)))
    end

    return results
end

export load_and_infer, replay_snapshots

end
```

`PINNSettings` import is gone. `settings` parameter is gone from both functions. No `.bin` filtering. No version branching.

### Step 5: Clean up stale references

**File:** `utils/training_schemes.jl` — line 20

Remove the now-unnecessary `safetensors_utils.jl` include (it's already included by `snapshot_utils.jl`):

```julia
# Line 20 — remove if redundant (snapshot_utils.jl already includes it)
# include("../utils/safetensors_utils.jl")
# using .SafeTensorSnapshots
```

**File:** `architectures/PINN.jl` — line 53-54

The `safetensors_utils.jl` include is still needed for `load_model` in the warm-start path. Keep it.

**File:** `src/main.jl` — line 56

Update CLI help text to remove stale `.bin` mention:

```julia
# Before:
help = "Path to .safetensors snapshot file for warm-start (.bin legacy snapshots are still supported)"

# After:
help = "Path to .safetensors model file for warm-start"
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `utils/safetensors_utils.jl` | Remove `save_safetensors_vector`, `load_safetensors_vector`, `load_snapshot_vector`, `DEFAULT_TENSOR_NAME`. Add 4 private helpers + `save_safetensors_model`, `load_safetensors_model`, `load_model`. Update exports. |
| `architectures/PINN.jl` | Lines 317-327: Replace `initialize_network` + snapshot overwrite pattern with `if snapshot_path` -> `load_model` else `initialize_network`. |
| `utils/training_schemes.jl` | Lines 87, 106: `save_safetensors_vector` -> `save_safetensors_model(p_current/coeff_net/seed)`. Possibly remove redundant `safetensors_utils.jl` include (L20-21). |
| `utils/snapshot_utils.jl` | Full rewrite: remove `PINNSettings` dependency, drop `settings` parameter from both functions, use `load_model`, remove `.bin` support. |
| `src/main.jl` | Line 56: Update CLI help text (remove `.bin` mention). |

## What Does NOT Change

- `loss_functions.jl`, `plugboard.jl`, `gpu_utils.jl`, `helper_funcs.jl` — untouched
- `two_d_grid_search_hyperparameters.jl` — no safetensors calls
- Training loop, loss computation, evaluation — untouched
- Mini-batching / `EpochBatchIterator` — untouched
- `training_results.json` output format — unchanged
- `evaluate_solution` — unchanged
- `initialize_network` — unchanged, used for fresh starts only
- `PINNSettings` struct — unchanged

## What Is Removed Entirely

| Removed | Location |
|---------|----------|
| `save_safetensors_vector` | `safetensors_utils.jl` |
| `load_safetensors_vector` | `safetensors_utils.jl` |
| `load_snapshot_vector` | `safetensors_utils.jl` |
| `DEFAULT_TENSOR_NAME` | `safetensors_utils.jl` |
| `.bin` read/write logic | `safetensors_utils.jl` |
| `settings` parameter from `load_and_infer` | `snapshot_utils.jl` |
| `settings` parameter from `replay_snapshots` | `snapshot_utils.jl` |
| `.bin` filtering in `replay_snapshots` | `snapshot_utils.jl` |
| `PINNSettings` import/usage in `snapshot_utils.jl` | `snapshot_utils.jl` |
| Version-detection / format branching in warm-start | `PINN.jl` |
| CLI help text referencing `.bin` | `src/main.jl` |

## Verification

1. **Roundtrip:** `save_safetensors_model` -> `load_model` -> identical outputs on the same input
2. **Self-containedness:** `load_model("model.safetensors")` works with zero external config — no settings, no architecture spec
3. **Checkpointing:** Short training produces v2 `.safetensors` with per-layer tensors, loadable by `load_model`
4. **Warm-start:** `--resume path/to/checkpoint.safetensors` resumes correctly — architecture matches the original run
5. **Snapshot replay:** `replay_snapshots(run_dir, ode_matrix)` works without passing `settings`
6. **Fresh start:** Training without `--resume` still works — `initialize_network` path untouched
7. **nn-viewer:** `load_and_infer(path, ode_matrix)` works with just 2 arguments (no settings)
