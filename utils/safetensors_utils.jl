module SafeTensorSnapshots

using JSON
using ComponentArrays
using Lux
using Serialization
import Random

function _write_u64_le(io, value::UInt64)
  for shift in 0:8:56
    write(io, UInt8((value >> shift) & 0xff))
  end
end

function _read_u64_le(bytes::Vector{UInt8})
  length(bytes) >= 8 || error("Invalid safetensors file: missing 8-byte header length")
  value = UInt64(0)
  for i in 1:8
    value |= UInt64(bytes[i]) << (8 * (i - 1))
  end
  return value
end

# Recursively collect leaf tensor names in ComponentArray natural axis order.
function _collect_leaf_names(x, prefix="")
  names = String[]
  if x isa ComponentArray
    for name in string.(propertynames(x))
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
    act_name = if act === Lux.σ || string(act) == "σ" || string(act) == "sigmoid"
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
      "in" => layer.in_dims,
      "out" => layer.out_dims,
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
      Lux.σ
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
    f32_arr = Float32.(Array(arr)) # Array() handles GPU->CPU
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

# ===========================================================================
# Raw checkpoint format (Julia Serialization) — fast native save/load.
# Training uses this format for checkpoints and resume; safetensors is only
# produced by the explicit `convert_to_safetensors` export step.
# ===========================================================================

# Flat CPU Float32 vector of the ComponentArray's underlying data, handling
# GPU-backed arrays transparently (Array(::CuArray) copies to host).
function _flat_cpu(p_ca)
  p_ca isa ComponentArray || (p_ca = ComponentArray(p_ca))
  flat = getdata(p_ca)
  return Vector{Float32}(Array(flat))
end

"""
    save_checkpoint(path, p_ca, coeff_net, seed;
                    representation=:power_series, iteration=nothing,
                    extra_metadata=Dict())

Write model weights in the native raw `.checkpoint` format (Julia Serialization).
Stores the flat CPU Float32 weights plus architecture/seed/representation metadata.
Fast to load — no JSON parse, no per-layer name traversal, no chain rebuild.
"""
function save_checkpoint(path::AbstractString, p_ca, coeff_net, seed::Int;
                         representation=:power_series, iteration=nothing,
                         extra_metadata::Dict=Dict{String,Any}())
  architecture = _extract_architecture(coeff_net)

  payload = Dict{String,Any}(
    "format" => "polymathjr-pinns-checkpoint",
    "version" => 1,
    "architecture" => architecture,
    "seed" => seed,
    "representation" => String(representation),
    "iteration" => iteration,
    "flat" => _flat_cpu(p_ca),
  )
  for (k, v) in extra_metadata
    payload[k] = v
  end

  mkpath(dirname(path))
  serialize(path, payload)
  return path
end

"""
    load_checkpoint(path) -> (coeff_net, p_ca, st, metadata)

Load a raw `.checkpoint` file. Rebuilds the chain from embedded architecture,
regenerates the Lux state from the saved seed, and reconstructs the ComponentArray
from the flat weights. Returns a ready-to-use model.
"""
function load_checkpoint(path::AbstractString)
  payload = deserialize(path)
  get(payload, "version", 1) == 1 || error("Unsupported checkpoint version: $(get(payload, "version", nothing))")

  architecture = payload["architecture"]
  seed = payload["seed"]
  coeff_net = _build_chain(architecture)

  rng = Random.default_rng()
  Random.seed!(rng, seed)
  p_template, st = Lux.setup(rng, coeff_net)
  p_template_ca = ComponentArray(p_template)

  p_ca = ComponentArray(Vector{Float32}(payload["flat"]), getaxes(p_template_ca))

  metadata = Dict{String,Any}(
    "format" => get(payload, "format", "polymathjr-pinns-checkpoint"),
    "version" => get(payload, "version", 1),
    "seed" => seed,
    "representation" => get(payload, "representation", "power_series"),
    "iteration" => get(payload, "iteration", nothing),
  )
  for (k, v) in payload
    k in ("architecture", "seed", "format", "version", "representation", "iteration", "flat") && continue
    metadata[k] = v
  end

  return coeff_net, p_ca, st, metadata
end

"""
    convert_to_safetensors(checkpoint_path, safetensors_path) -> safetensors_path

Explicit conversion step: load a raw `.checkpoint` and write it as a shared
Hugging Face `.safetensors` file.
"""
function convert_to_safetensors(checkpoint_path::AbstractString, safetensors_path::AbstractString)
  coeff_net, p_ca, _, metadata = load_checkpoint(checkpoint_path)
  seed = metadata["seed"]

  extra = Dict{String,Any}(
    "representation" => get(metadata, "representation", "power_series"),
    "objective_components" => "pde + supervised",
    "diagnostic_components" => "bc",
  )
  for (k, v) in metadata
    k in ("seed", "architecture", "format", "version") && continue
    extra[string(k)] = v
  end

  return save_safetensors_model(safetensors_path, p_ca, coeff_net, seed; extra_metadata=extra)
end

"""
    load_any_model(path) -> (coeff_net, p_ca, st, metadata)

Dispatch by extension: `.checkpoint` loads via the raw path (fast), anything else
falls back to the safetensors path (for old `.safetensors` files).
"""
function load_any_model(path::AbstractString)
  if endswith(lowercase(path), ".checkpoint")
    return load_checkpoint(path)
  else
    return load_model(path)
  end
end

export save_safetensors_model, load_safetensors_model, load_model,
       save_checkpoint, load_checkpoint, convert_to_safetensors, load_any_model

end
