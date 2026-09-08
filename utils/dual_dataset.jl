module DualDataset

using Random
using SHA
using JSON
using Mmap

export AbstractODESplit, CanonicalODESplit, MaterializedODESplit,
       dataset_id, dataset_manifest, materialize_split, load_materialized_split,
       example_at, parameters_at, batch_items, family_batch_items,
       evaluation_batch_items,
       family_size, batches_per_epoch

const GENERATOR_VERSION = "dual-trace-determinant-v1"
const REGIONS = [:saddle, :stable_node, :unstable_node,
                 :stable_spiral, :unstable_spiral, :center]

alpha_from_tau_delta(tau, delta) =
  reshape(Float32[delta, -tau, 1.0f0], 3, 1)

abstract type AbstractODESplit end

struct CanonicalODESplit <: AbstractODESplit
  name::Symbol
  size::Int
  seed::UInt64
  shuffle_seed::UInt64
  N::Int
  regions::Vector{Symbol}
end

"""
An immutable, copyable split backed by three raw little-endian arrays.

`tau[i]`, `delta[i]`, and `region_codes[i]` are the complete canonical ODE
record at index `i`. Power-series and eigenvalue targets are deliberately
derived from that same stored ODE rather than persisted as two independent
datasets.
"""
struct MaterializedODESplit <: AbstractODESplit
  name::Symbol
  size::Int
  seed::UInt64
  shuffle_seed::UInt64
  N::Int
  regions::Vector{Symbol}
  id::String
  root::String
  tau::Vector{Float32}
  delta::Vector{Float32}
  region_codes::Vector{UInt8}
end

function CanonicalODESplit(name::Symbol, size::Int, seed::Integer, N::Int;
                           shuffle_seed::Integer=seed,
                           regions=REGIONS)
  size > 0 || error("split size must be positive")
  isempty(regions) && error("at least one region is required")
  all(r -> r in REGIONS, regions) || error("unknown trace-determinant region")
  CanonicalODESplit(name, size, UInt64(seed), UInt64(shuffle_seed), N, Symbol[regions...])
end

dataset_id(s::CanonicalODESplit) = bytes2hex(sha256(join((
  GENERATOR_VERSION, String(s.name), string(s.size), string(s.seed),
  string(s.N), join(String.(s.regions), ",")), "|")))
dataset_id(s::MaterializedODESplit) = s.id

@inline function splitmix64(x::UInt64)
  z = x + 0x9e3779b97f4a7c15
  z = xor(z, z >> 30) * 0xbf58476d1ce4e5b9
  z = xor(z, z >> 27) * 0x94d049bb133111eb
  return xor(z, z >> 31)
end

@inline unitfloat(seed::UInt64, index::Int, lane::Int) =
  Float32((splitmix64(xor(seed, UInt64(index), UInt64(lane) << 32)) >> 40) / 16777216.0)

function parameters_for(region::Symbol, u1::Float32, u2::Float32)
  if region === :saddle
    return (4f0 * u1 - 2f0, -(0.01f0 + 1.99f0 * u2))
  elseif region === :stable_node || region === :unstable_node
    sign = region === :stable_node ? -1f0 : 1f0
    r1 = sign * (0.01f0 + 0.99f0 * u1)
    r2 = sign * (0.01f0 + 0.99f0 * u2)
    return (r1 + r2, r1 * r2)
  elseif region === :stable_spiral || region === :unstable_spiral
    sign = region === :stable_spiral ? -1f0 : 1f0
    mu = sign * (0.01f0 + 0.99f0 * u1)
    omega = 0.01f0 + 0.99f0 * u2
    return (2f0 * mu, mu * mu + omega * omega)
  elseif region === :center
    omega = 0.01f0 + 1.40f0 * u1
    return (0f0, omega * omega)
  end
  error("unsupported region: $region")
end

function parameters_at(s::CanonicalODESplit, index::Int)
  1 <= index <= s.size || throw(BoundsError(1:s.size, index))
  region = s.regions[mod1(index, length(s.regions))]
  tau, delta = parameters_for(region, unitfloat(s.seed, index, 1), unitfloat(s.seed, index, 2))
  return region, tau, delta
end

function parameters_at(s::MaterializedODESplit, index::Int)
  1 <= index <= s.size || throw(BoundsError(1:s.size, index))
  code = Int(s.region_codes[index])
  1 <= code <= length(s.regions) || error("invalid region code $code at canonical index $index")
  return s.regions[code], s.tau[index], s.delta[index]
end

function example_at(s::AbstractODESplit, index::Int)
  _, tau, delta = parameters_at(s, index)
  alpha = alpha_from_tau_delta(tau, delta)
  series = Vector{Float32}(undef, s.N + 1)
  series[1] = 1f0
  s.N >= 1 && (series[2] = 0f0)
  for n in 1:(s.N - 1)
    series[n + 2] = tau * series[n + 1] - delta * series[n]
  end
  return alpha => series
end

batches_per_epoch(s::AbstractODESplit, batch_size::Int) = cld(s.size, batch_size)

function family_size(s::AbstractODESplit, region::Symbol)
  position = findfirst(==(region), s.regions)
  position === nothing && return 0
  s.size < position && return 0
  return fld(s.size - position, length(s.regions)) + 1
end

function batch_items(s::AbstractODESplit, epoch::Int, batch::Int, batch_size::Int)
  nbatches = batches_per_epoch(s, batch_size)
  1 <= batch <= nbatches || throw(BoundsError(1:nbatches, batch))
  # An affine permutation gives a complete, allocation-free shuffle each epoch.
  # D-1 is always coprime to D, so every canonical index appears exactly once.
  multiplier = s.size == 1 ? 1 : s.size - 1
  offset = Int(mod(splitmix64(xor(s.shuffle_seed, UInt64(epoch))), UInt64(s.size)))
  first_position = (batch - 1) * batch_size
  count = min(batch_size, s.size - first_position)
  items = Vector{Pair{Any,Any}}(undef, count)
  for j in 1:count
    position = first_position + j - 1
    index = mod(multiplier * position + offset, s.size) + 1
    items[j] = example_at(s, index)
  end
  return items
end

function family_batch_items(s::AbstractODESplit, region::Symbol, epoch::Int,
                            batch::Int, batch_size::Int)
  region_position = findfirst(==(region), s.regions)
  region_position === nothing && error("region $region is not present in split $(s.name)")
  count_total = family_size(s, region)
  nbatches = cld(count_total, batch_size)
  1 <= batch <= nbatches || throw(BoundsError(1:nbatches, batch))
  multiplier = count_total == 1 ? 1 : count_total - 1
  offset = Int(mod(splitmix64(xor(s.shuffle_seed, UInt64(epoch), UInt64(region_position))), UInt64(count_total)))
  first_position = (batch - 1) * batch_size
  count = min(batch_size, count_total - first_position)
  items = Vector{Pair{Any,Any}}(undef, count)
  for j in 1:count
    position = first_position + j - 1
    family_index = mod(multiplier * position + offset, count_total)
    canonical_index = region_position + family_index * length(s.regions)
    items[j] = example_at(s, canonical_index)
  end
  return items
end

"""
Return a stable, unshuffled evaluation batch. With `region=nothing`, batches
cover canonical indices 1:size. With a region, they cover only that family's
canonical subsequence. This ordering is recorded and reused for validation and
test inference.
"""
function evaluation_batch_items(s::AbstractODESplit, batch::Int, batch_size::Int;
                                region::Union{Symbol,Nothing}=nothing)
  batch_size > 0 || error("batch_size must be positive")
  if region === nothing
    total = s.size
    first_index = (batch - 1) * batch_size + 1
    1 <= first_index <= total || throw(BoundsError(1:cld(total, batch_size), batch))
    last_index = min(first_index + batch_size - 1, total)
    return [example_at(s, index) for index in first_index:last_index]
  end

  region_position = findfirst(==(region), s.regions)
  region_position === nothing && error("region $region is not present in split $(s.name)")
  total = family_size(s, region)
  first_family_index = (batch - 1) * batch_size + 1
  1 <= first_family_index <= total ||
    throw(BoundsError(1:cld(total, batch_size), batch))
  last_family_index = min(first_family_index + batch_size - 1, total)
  return [example_at(s, region_position + (family_index - 1) * length(s.regions))
          for family_index in first_family_index:last_family_index]
end

function _sha256_file(path::AbstractString)
  open(path, "r") do io
    return bytes2hex(sha256(io))
  end
end

function _file_entry(path::AbstractString, dtype::String)
  Dict(
    "path" => basename(path),
    "dtype" => dtype,
    "bytes" => filesize(path),
    "sha256" => _sha256_file(path),
  )
end

function dataset_manifest(s::CanonicalODESplit, split_root::AbstractString)
  Dict(
    "format" => "polymathjr-portable-ode-split",
    "format_version" => 1,
    "generator_version" => GENERATOR_VERSION,
    "name" => String(s.name),
    "size" => s.size,
    "seed" => string(s.seed),
    "shuffle_seed" => string(s.shuffle_seed),
    "dataset_id" => dataset_id(s),
    "regions" => String.(s.regions),
    "power_series_degree" => s.N,
    "canonical_index" => "one-based row number",
    "endianness" => "little",
    "record_semantics" => "tau and delta define u'' - tau*u' + delta*u = 0 with u(0)=1, u'(0)=0",
    "files" => Dict(
      "tau" => _file_entry(joinpath(split_root, "tau.f32"), "Float32"),
      "delta" => _file_entry(joinpath(split_root, "delta.f32"), "Float32"),
      "region" => _file_entry(joinpath(split_root, "region.u8"), "UInt8"),
    ),
  )
end

function _write_materialized_arrays(s::CanonicalODESplit, split_root::AbstractString;
                                    chunk_size::Int=1_000_000)
  chunk_size > 0 || error("chunk_size must be positive")
  tau_path = joinpath(split_root, "tau.f32")
  delta_path = joinpath(split_root, "delta.f32")
  region_path = joinpath(split_root, "region.u8")
  open(tau_path, "w") do tau_io
    open(delta_path, "w") do delta_io
      open(region_path, "w") do region_io
        first_index = 1
        while first_index <= s.size
          count = min(chunk_size, s.size - first_index + 1)
          taus = Vector{Float32}(undef, count)
          deltas = Vector{Float32}(undef, count)
          codes = Vector{UInt8}(undef, count)
          for offset in 0:(count - 1)
            index = first_index + offset
            region, tau, delta = parameters_at(s, index)
            taus[offset + 1] = tau
            deltas[offset + 1] = delta
            codes[offset + 1] = UInt8(findfirst(==(region), s.regions))
          end
          write(tau_io, htol.(taus))
          write(delta_io, htol.(deltas))
          write(region_io, codes)
          first_index += count
        end
      end
    end
  end
end

"""
    materialize_split(split, split_root; force=false) -> MaterializedODESplit

Persist the canonical ODE population once. Existing complete splits are
verified and reused; an incomplete directory is rejected unless `force=true`.
"""
function materialize_split(s::CanonicalODESplit, split_root::AbstractString;
                           force::Bool=false, chunk_size::Int=1_000_000,
                           verify::Bool=true)
  manifest_path = joinpath(split_root, "manifest.json")
  if isfile(manifest_path) && !force
    return load_materialized_split(split_root, s; verify=verify)
  end
  if isdir(split_root) && !isempty(readdir(split_root)) && !force
    error("Refusing to reuse incomplete materialized split at $split_root; use a new study directory")
  end
  mkpath(split_root)
  for filename in ("tau.f32", "delta.f32", "region.u8", "manifest.json")
    path = joinpath(split_root, filename)
    isfile(path) && rm(path; force=true)
  end
  _write_materialized_arrays(s, split_root; chunk_size=chunk_size)
  manifest = dataset_manifest(s, split_root)
  open(manifest_path, "w") do io
    JSON.print(io, manifest, 2)
  end
  return load_materialized_split(split_root, s; verify=verify)
end

function _mmap_vector(path::AbstractString, ::Type{T}, count::Int) where {T}
  open(path, "r") do io
    return Mmap.mmap(io, Vector{T}, count)
  end
end

"""Load and optionally checksum a portable split, asserting its expected recipe."""
function load_materialized_split(split_root::AbstractString,
                                 expected::Union{CanonicalODESplit,Nothing}=nothing;
                                 verify::Bool=true)
  manifest_path = joinpath(split_root, "manifest.json")
  isfile(manifest_path) || error("Missing materialized dataset manifest: $manifest_path")
  manifest = JSON.parsefile(manifest_path)
  get(manifest, "format", nothing) == "polymathjr-portable-ode-split" ||
    error("Unsupported dataset format in $manifest_path")
  get(manifest, "format_version", nothing) == 1 ||
    error("Unsupported dataset format version in $manifest_path")

  name = Symbol(manifest["name"])
  size = Int(manifest["size"])
  seed = parse(UInt64, manifest["seed"])
  shuffle_seed = parse(UInt64, manifest["shuffle_seed"])
  N = Int(manifest["power_series_degree"])
  regions = Symbol.(manifest["regions"])
  id = String(manifest["dataset_id"])
  if expected !== nothing
    id == dataset_id(expected) || error("Dataset ID mismatch at $split_root")
    name == expected.name || error("Dataset split name mismatch at $split_root")
    size == expected.size || error("Dataset size mismatch at $split_root")
    shuffle_seed == expected.shuffle_seed || error("Dataset shuffle seed mismatch at $split_root")
  end

  paths = Dict(key => joinpath(split_root, manifest["files"][key]["path"])
               for key in ("tau", "delta", "region"))
  expected_bytes = Dict("tau" => 4size, "delta" => 4size, "region" => size)
  for key in keys(paths)
    isfile(paths[key]) || error("Missing materialized dataset file: $(paths[key])")
    filesize(paths[key]) == expected_bytes[key] ||
      error("Unexpected byte count for $(paths[key])")
    if verify
      actual = _sha256_file(paths[key])
      actual == manifest["files"][key]["sha256"] ||
        error("Checksum mismatch for $(paths[key])")
    end
  end

  tau_raw = _mmap_vector(paths["tau"], Float32, size)
  delta_raw = _mmap_vector(paths["delta"], Float32, size)
  tau = ENDIAN_BOM == 0x04030201 ? tau_raw : ltoh.(tau_raw)
  delta = ENDIAN_BOM == 0x04030201 ? delta_raw : ltoh.(delta_raw)
  region_codes = _mmap_vector(paths["region"], UInt8, size)
  return MaterializedODESplit(name, size, seed, shuffle_seed, N, regions, id,
                              String(split_root), tau, delta, region_codes)
end

end
