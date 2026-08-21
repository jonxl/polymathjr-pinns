# =============================================================================
# Test: Raw Checkpoint Format (Julia Serialization) + safetensors conversion
# =============================================================================
# Verifies:
#   1. save_checkpoint -> load_checkpoint roundtrip (identical weights + outputs)
#   2. Self-containedness: load_checkpoint works with just a file path
#   3. Architecture + seed + representation roundtrip through metadata
#   4. convert_to_safetensors produces a valid v2 safetensors file
#   5. load_any_model dispatches on extension (.checkpoint vs .safetensors)
# =============================================================================

using Test
using Random
using Lux
using ComponentArrays

include("../utils/safetensors_utils.jl")
using .SafeTensorSnapshots

function build_net(in_dim::Int, hidden::Int, out_dim::Int; seed::Int=42)
  chain = Lux.Chain(
    Lux.Dense(in_dim, hidden, Lux.σ),
    Lux.Dense(hidden, hidden, Lux.σ),
    Lux.Dense(hidden, out_dim),
  )
  rng = Random.default_rng()
  Random.seed!(rng, seed)
  p, st = Lux.setup(rng, chain)
  return chain, ComponentArray(p), st, seed
end

@testset "Raw checkpoint roundtrip" begin
  chain, p_ca, st, seed = build_net(3, 5, 2; seed=123)

  tmpfile = tempname() * ".checkpoint"
  try
    # Save
    result_path = save_checkpoint(tmpfile, p_ca, chain, seed;
                                  representation=:power_series, iteration=42)
    @test result_path == tmpfile
    @test isfile(tmpfile)

    # Load
    chain2, p_ca2, st2, meta = load_checkpoint(tmpfile)
    @test meta["format"] == "polymathjr-pinns-checkpoint"
    @test meta["version"] == 1
    @test meta["seed"] == 123
    @test meta["representation"] == "power_series"
    @test meta["iteration"] == 42
    @test chain2 isa Lux.Chain
    @test p_ca2 isa ComponentArray

    # Weights match exactly
    @test getdata(p_ca2) == getdata(p_ca)

    # Inference matches
    x = Float32[0.5, -1.0, 2.0]
    @test first(chain2(x, p_ca2, st2)) ≈ first(chain(x, p_ca, st))
    @info "✓ Raw checkpoint roundtrip identical weights + outputs"
  finally
    isfile(tmpfile) && rm(tmpfile)
  end
end

@testset "load_any_model extension dispatch" begin
  chain, p_ca, st, seed = build_net(3, 5, 2; seed=7)

  ckpt = tempname() * ".checkpoint"
  stf = tempname() * ".safetensors"
  try
    save_checkpoint(ckpt, p_ca, chain, seed)
    convert_to_safetensors(ckpt, stf)

    # .checkpoint -> raw path
    c1, p1, _, m1 = load_any_model(ckpt)
    @test m1["format"] == "polymathjr-pinns-checkpoint"

    # .safetensors -> safetensors path
    c2, p2, _, m2 = load_any_model(stf)
    @test m2["format"] == "polymathjr-pinns-model"

    # Both reconstruct identical weights
    @test getdata(p1) == getdata(p2)
    @info "✓ load_any_model dispatches correctly by extension"
  finally
    isfile(ckpt) && rm(ckpt)
    isfile(stf) && rm(stf)
  end
end

@testset "convert_to_safetensors produces valid v2" begin
  chain, p_ca, st, seed = build_net(3, 4, 2; seed=99)

  ckpt = tempname() * ".checkpoint"
  stf = tempname() * ".safetensors"
  try
    save_checkpoint(ckpt, p_ca, chain, seed; representation=:eigenvalue)
    convert_to_safetensors(ckpt, stf)

    meta, tensors = load_safetensors_model(stf)
    @test meta["format"] == "polymathjr-pinns-model"
    @test meta["version"] == 2
    @test meta["seed"] == 99
    @test meta["representation"] == "eigenvalue"
    @test length(meta["architecture"]) == 3
    @test length(tensors) == 6  # 3 layers × 2 params
    @info "✓ conversion emits valid v2 safetensors with representation metadata"
  finally
    isfile(ckpt) && rm(ckpt)
    isfile(stf) && rm(stf)
  end
end

@info "ALL RAW CHECKPOINT TESTS PASSED ✓"
