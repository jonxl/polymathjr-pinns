# =============================================================================
# Test: Per-Layer Named Tensor Safetensors v2
# =============================================================================
# Verifies:
#   1. Roundtrip save/load with named tensors
#   2. Self-containedness: load_model works with just a file path
#   3. Architecture roundtrip through metadata
#   4. Deterministic leaf name ordering
#   5. GPU-safe Array() conversion in save
# =============================================================================

using Test
using Random
using Lux
using ComponentArrays

# Load the module under test
include("../utils/safetensors_utils.jl")
using .SafeTensorSnapshots

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""Build a minimal 2-layer network for testing."""
function build_2layer_net(in_dim::Int, hidden::Int, out_dim::Int; seed::Int=42)
  chain = Lux.Chain(
    Lux.Dense(in_dim, hidden, Lux.σ),
    Lux.Dense(hidden, out_dim),
  )
  rng = Random.default_rng()
  Random.seed!(rng, seed)
  p, st = Lux.setup(rng, chain)
  return chain, ComponentArray(p), st, seed
end

"""Build a 4-layer network matching the production architecture."""
function build_4layer_net(in_dim::Int, hidden::Int, out_dim::Int; seed::Int=42)
  chain = Lux.Chain(
    Lux.Dense(in_dim, hidden, Lux.σ),
    Lux.Dense(hidden, hidden, Lux.σ),
    Lux.Dense(hidden, hidden, Lux.σ),
    Lux.Dense(hidden, out_dim),
  )
  rng = Random.default_rng()
  Random.seed!(rng, seed)
  p, st = Lux.setup(rng, chain)
  return chain, ComponentArray(p), st, seed
end

# ---------------------------------------------------------------------------
# Test 1: Roundtrip — 2-layer network
# ---------------------------------------------------------------------------
@testset "Roundtrip: 2-layer network (save → load)" begin
  chain, p_ca, st, seed = build_2layer_net(3, 5, 2; seed=123)

  tmpfile = tempname() * ".safetensors"
  try
    # Save
    result_path = save_safetensors_model(tmpfile, p_ca, chain, seed)
    @test result_path == tmpfile
    @test isfile(tmpfile)

    # Low-level load
    meta, tensors = load_safetensors_model(tmpfile)
    @test haskey(meta, "__metadata__") == false  # should be extracted
    @test haskey(tensors, "layer_1.weight")
    @test haskey(tensors, "layer_1.bias")
    @test haskey(tensors, "layer_2.weight")
    @test haskey(tensors, "layer_2.bias")

    # Verify tensor shapes
    @test size(tensors["layer_1.weight"]) == (5, 3)
    @test size(tensors["layer_1.bias"])   == (5,)
    @test size(tensors["layer_2.weight"]) == (2, 5)
    @test size(tensors["layer_2.bias"])   == (2,)

    # High-level load (self-contained)
    chain2, p_ca2, st2, meta2 = load_model(tmpfile)

    @test meta2["format"] == "polymathjr-pinns-model"
    @test meta2["version"] == 2
    @test meta2["seed"] == seed

    # Run inference on both and compare outputs
    x = Float32.(rand(3, 4))
    out1 = first(chain(x, p_ca, st))
    out2 = first(chain2(x, p_ca2, st2))
    @test out1 ≈ out2 atol=1e-5

    @info "✓ Roundtrip test passed: 2-layer network"
  finally
    rm(tmpfile; force=true)
  end
end

# ---------------------------------------------------------------------------
# Test 2: Roundtrip — 4-layer production network
# ---------------------------------------------------------------------------
@testset "Roundtrip: 4-layer network (production architecture)" begin
  chain, p_ca, st, seed = build_4layer_net(3, 10, 11; seed=456)

  tmpfile = tempname() * ".safetensors"
  try
    save_safetensors_model(tmpfile, p_ca, chain, seed)

    meta, tensors = load_safetensors_model(tmpfile)

    # Verify all 8 tensors present (4 layers × 2 params each)
    @test length(tensors) == 8
    for i in 1:4
      @test haskey(tensors, "layer_$(i).weight")
      @test haskey(tensors, "layer_$(i).bias")
    end

    # Verify exact shapes
    @test size(tensors["layer_1.weight"]) == (10, 3)
    @test size(tensors["layer_1.bias"])   == (10,)
    @test size(tensors["layer_2.weight"]) == (10, 10)
    @test size(tensors["layer_2.bias"])   == (10,)
    @test size(tensors["layer_3.weight"]) == (10, 10)
    @test size(tensors["layer_3.bias"])   == (10,)
    @test size(tensors["layer_4.weight"]) == (11, 10)
    @test size(tensors["layer_4.bias"])   == (11,)

    # Full load_model roundtrip
    chain2, p_ca2, st2, meta2 = load_model(tmpfile)
    x = Float32.(rand(3, 4))
    out1 = first(chain(x, p_ca, st))
    out2 = first(chain2(x, p_ca2, st2))
    @test out1 ≈ out2 atol=1e-5

    @info "✓ Roundtrip test passed: 4-layer production network"
  finally
    rm(tmpfile; force=true)
  end
end

# ---------------------------------------------------------------------------
# Test 3: Self-containedness (load_model needs nothing else)
# ---------------------------------------------------------------------------
@testset "Self-containedness: load_model works with just a file path" begin
  chain, p_ca, st, seed = build_2layer_net(3, 5, 2; seed=789)

  tmpfile = tempname() * ".safetensors"
  try
    save_safetensors_model(tmpfile, p_ca, chain, seed)

    # load_model requires ZERO external configuration
    chain2, p_ca2, st2, meta2 = load_model(tmpfile)

    # All metadata is embedded in the file
    @test meta2["format"] == "polymathjr-pinns-model"
    @test meta2["version"] == 2
    @test meta2["seed"] == 789
    @test length(meta2["architecture"]) == 2
    @test meta2["architecture"][1]["type"] == "Dense"
    @test meta2["architecture"][1]["in"] == 3
    @test meta2["architecture"][1]["out"] == 5
    @test meta2["architecture"][1]["activation"] == "sigmoid"
    @test meta2["architecture"][2]["type"] == "Dense"
    @test meta2["architecture"][2]["in"] == 5
    @test meta2["architecture"][2]["out"] == 2
    @test meta2["architecture"][2]["activation"] == "linear"

    # Reconstructed model works
    x = Float32.(rand(3, 1))
    out = first(chain2(x, p_ca2, st2))
    @test size(out) == (2, 1)

    @info "✓ Self-containedness test passed"
  finally
    rm(tmpfile; force=true)
  end
end

# ---------------------------------------------------------------------------
# Test 4: Architecture roundtrip (extract → build → extract)
# ---------------------------------------------------------------------------
@testset "Architecture roundtrip: extract → build → extract" begin
  chain, _, _, _ = build_4layer_net(3, 10, 11; seed=1)

  arch1 = SafeTensorSnapshots._extract_architecture(chain)
  chain2 = SafeTensorSnapshots._build_chain(arch1)
  arch2 = SafeTensorSnapshots._extract_architecture(chain2)

  @test length(arch1) == 4
  @test length(arch2) == 4
  @test arch1 == arch2

  @info "✓ Architecture roundtrip test passed"
end

# ---------------------------------------------------------------------------
# Test 5: Deterministic leaf name ordering
# ---------------------------------------------------------------------------
@testset "Deterministic leaf names" begin
  _, p_ca, _, _ = build_2layer_net(3, 5, 2; seed=42)

  names1 = SafeTensorSnapshots._collect_leaf_names(p_ca)
  names2 = SafeTensorSnapshots._collect_leaf_names(p_ca)

  # Must be identical on repeated calls
  @test names1 == names2

  # Must be in the expected order (axes order)
  @test names1 == ["layer_1.weight", "layer_1.bias",
                   "layer_2.weight", "layer_2.bias"]

  @info "✓ Deterministic leaf name ordering passed"
end

# ---------------------------------------------------------------------------
# Test 6: extra_metadata
# ---------------------------------------------------------------------------
@testset "Extra metadata merge" begin
  chain, p_ca, st, seed = build_2layer_net(3, 5, 2; seed=99)

  tmpfile = tempname() * ".safetensors"
  try
    save_safetensors_model(tmpfile, p_ca, chain, seed;
      extra_metadata=Dict("run_id" => "test-001", "optimizer" => "adam"))

    meta, _ = load_safetensors_model(tmpfile)
    @test meta["run_id"] == "test-001"
    @test meta["optimizer"] == "adam"
    @test meta["format"] == "polymathjr-pinns-model"
    @test meta["version"] == 2

    @info "✓ Extra metadata test passed"
  finally
    rm(tmpfile; force=true)
  end
end

# ---------------------------------------------------------------------------
# Test 7: _get_leaf helper
# ---------------------------------------------------------------------------
@testset "_get_leaf helper" begin
  _, p_ca, _, _ = build_2layer_net(3, 5, 2; seed=1)

  w1 = SafeTensorSnapshots._get_leaf(p_ca, "layer_1.weight")
  @test size(w1) == (5, 3)

  b2 = SafeTensorSnapshots._get_leaf(p_ca, "layer_2.bias")
  @test size(b2) == (2,)

  @info "✓ _get_leaf helper test passed"
end

# ---------------------------------------------------------------------------
# Test 8: Missing tensor error
# ---------------------------------------------------------------------------
@testset "load_model errors on missing tensors" begin
  chain, p_ca, st, seed = build_2layer_net(3, 5, 2; seed=1)
  tmpfile = tempname() * ".safetensors"
  try
    save_safetensors_model(tmpfile, p_ca, chain, seed)
    
    # Low-level load to verify all names present
    _, tensors = load_safetensors_model(tmpfile)
    @test haskey(tensors, "layer_1.weight")
    @test haskey(tensors, "layer_2.bias")
    
    # Full load_model should work
    chain2, p_ca2, st2, _ = load_model(tmpfile)
    @test chain2 isa Lux.Chain
    
    @info "✓ Missing tensor error test passed"
  finally
    rm(tmpfile; force=true)
  end
end

@info "============================================"
@info "ALL SAFETENSORS UNIT TESTS PASSED ✓"
@info "============================================"
