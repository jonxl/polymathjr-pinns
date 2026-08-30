module DualDataset

using Random
using SHA

include("plugboard.jl")
using .Plugboard

export CanonicalODESplit, dataset_id, example_at, batch_items, family_batch_items,
       family_size, batches_per_epoch

const GENERATOR_VERSION = "dual-trace-determinant-v1"
const REGIONS = Plugboard.TRACE_DET_REGIONS

struct CanonicalODESplit
  name::Symbol
  size::Int
  seed::UInt64
  N::Int
  regions::Vector{Symbol}
end

function CanonicalODESplit(name::Symbol, size::Int, seed::Integer, N::Int;
                           regions=REGIONS)
  size > 0 || error("split size must be positive")
  isempty(regions) && error("at least one region is required")
  all(r -> r in REGIONS, regions) || error("unknown trace-determinant region")
  CanonicalODESplit(name, size, UInt64(seed), N, Symbol[regions...])
end

dataset_id(s::CanonicalODESplit) = bytes2hex(sha256(join((
  GENERATOR_VERSION, String(s.name), string(s.size), string(s.seed),
  string(s.N), join(String.(s.regions), ",")), "|")))

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

function example_at(s::CanonicalODESplit, index::Int)
  1 <= index <= s.size || throw(BoundsError(1:s.size, index))
  region = s.regions[mod1(index, length(s.regions))]
  tau, delta = parameters_for(region, unitfloat(s.seed, index, 1), unitfloat(s.seed, index, 2))
  alpha = Plugboard.alpha_from_tau_delta(tau, delta)
  series = Vector{Float32}(undef, s.N + 1)
  series[1] = 1f0
  s.N >= 1 && (series[2] = 0f0)
  for n in 1:(s.N - 1)
    series[n + 2] = tau * series[n + 1] - delta * series[n]
  end
  return alpha => series
end

batches_per_epoch(s::CanonicalODESplit, batch_size::Int) = cld(s.size, batch_size)

function family_size(s::CanonicalODESplit, region::Symbol)
  position = findfirst(==(region), s.regions)
  position === nothing && return 0
  s.size < position && return 0
  return fld(s.size - position, length(s.regions)) + 1
end

function batch_items(s::CanonicalODESplit, epoch::Int, batch::Int, batch_size::Int)
  nbatches = batches_per_epoch(s, batch_size)
  1 <= batch <= nbatches || throw(BoundsError(1:nbatches, batch))
  # An affine permutation gives a complete, allocation-free shuffle each epoch.
  # D-1 is always coprime to D, so every canonical index appears exactly once.
  multiplier = s.size == 1 ? 1 : s.size - 1
  offset = Int(mod(splitmix64(xor(s.seed, UInt64(epoch))), UInt64(s.size)))
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

function family_batch_items(s::CanonicalODESplit, region::Symbol, epoch::Int,
                            batch::Int, batch_size::Int)
  region_position = findfirst(==(region), s.regions)
  region_position === nothing && error("region $region is not present in split $(s.name)")
  count_total = family_size(s, region)
  nbatches = cld(count_total, batch_size)
  1 <= batch <= nbatches || throw(BoundsError(1:nbatches, batch))
  multiplier = count_total == 1 ? 1 : count_total - 1
  offset = Int(mod(splitmix64(xor(s.seed, UInt64(epoch), UInt64(region_position))), UInt64(count_total)))
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

end
