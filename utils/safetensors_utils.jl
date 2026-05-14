module SafeTensorSnapshots

using JSON

const DEFAULT_TENSOR_NAME = "weights"

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

"""
    save_safetensors_vector(path, values; tensor_name="weights")

Write a single flattened Float32 tensor in Hugging Face safetensors format.
"""
function save_safetensors_vector(path::AbstractString, values; tensor_name::AbstractString=DEFAULT_TENSOR_NAME)
  weights = Float32.(vec(values))
  byte_len = length(weights) * sizeof(Float32)

  header = Dict{String,Any}(
    tensor_name => Dict(
      "dtype" => "F32",
      "shape" => [length(weights)],
      "data_offsets" => [0, byte_len],
    ),
    "__metadata__" => Dict(
      "format" => "polymathjr-pinns-snapshot",
    ),
  )

  header_bytes = Vector{UInt8}(codeunits(JSON.json(header)))
  mkpath(dirname(path))

  open(path, "w") do io
    _write_u64_le(io, UInt64(length(header_bytes)))
    write(io, header_bytes)
    write(io, weights)
  end

  return path
end

function load_safetensors_vector(path::AbstractString; tensor_name::AbstractString=DEFAULT_TENSOR_NAME)
  bytes = read(path)
  header_len = Int(_read_u64_le(bytes))
  header_start = 9
  header_end = 8 + header_len
  header_end <= length(bytes) || error("Invalid safetensors file: header exceeds file size")

  header = JSON.parse(String(bytes[header_start:header_end]))
  haskey(header, tensor_name) || error("Safetensors file does not contain tensor \"$tensor_name\"")

  tensor_info = header[tensor_name]
  tensor_info["dtype"] == "F32" || error("Unsupported tensor dtype: $(tensor_info["dtype"]). Expected F32.")

  offsets = tensor_info["data_offsets"]
  length(offsets) == 2 || error("Invalid safetensors file: expected two data offsets")

  data_start = header_end + 1 + Int(offsets[1])
  data_end = header_end + Int(offsets[2])
  data_start <= data_end <= length(bytes) || error("Invalid safetensors file: tensor data exceeds file size")

  data_bytes = bytes[data_start:data_end]
  length(data_bytes) % sizeof(Float32) == 0 || error("Invalid F32 tensor byte length")

  values = collect(reinterpret(Float32, data_bytes))
  expected_len = prod(Int.(tensor_info["shape"]))
  length(values) == expected_len || error("Safetensors shape does not match tensor data length")

  return values
end

"""
    load_snapshot_vector(path; tensor_name="weights")

Load current `.safetensors` snapshots, with legacy raw `.bin` support.
"""
function load_snapshot_vector(path::AbstractString; tensor_name::AbstractString=DEFAULT_TENSOR_NAME)
  if endswith(lowercase(path), ".safetensors")
    return load_safetensors_vector(path; tensor_name=tensor_name)
  elseif endswith(lowercase(path), ".bin")
    return collect(reinterpret(Float32, read(path)))
  else
    error("Unsupported snapshot format for $path. Expected .safetensors or legacy .bin.")
  end
end

export save_safetensors_vector, load_safetensors_vector, load_snapshot_vector

end
