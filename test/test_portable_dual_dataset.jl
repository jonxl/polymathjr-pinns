using Test
using JSON

include("../utils/dual_dataset.jl")
using .DualDataset

@testset "Portable dual-representation dataset" begin
  recipe = CanonicalODESplit(:train, 37, 10001, 20;
    shuffle_seed=40001,
    regions=[:saddle, :stable_node, :unstable_node,
             :stable_spiral, :unstable_spiral, :center])

  mktempdir() do temp_root
    split_root = joinpath(temp_root, "train")
    stored = materialize_split(recipe, split_root; chunk_size=7)

    @test dataset_id(stored) == dataset_id(recipe)
    @test stored.size == recipe.size
    @test filesize(joinpath(split_root, "tau.f32")) == 4 * recipe.size
    @test filesize(joinpath(split_root, "delta.f32")) == 4 * recipe.size
    @test filesize(joinpath(split_root, "region.u8")) == recipe.size

    manifest = JSON.parsefile(joinpath(split_root, "manifest.json"))
    @test manifest["format"] == "polymathjr-portable-ode-split"
    @test manifest["size"] == recipe.size
    @test all(length(manifest["files"][key]["sha256"]) == 64
              for key in ("tau", "delta", "region"))

    for index in 1:recipe.size
      @test parameters_at(stored, index) == parameters_at(recipe, index)
      generated = example_at(recipe, index)
      materialized = example_at(stored, index)
      @test first(materialized) == first(generated)
      @test last(materialized) == last(generated)
    end

    @test batch_items(stored, 3, 2, 8) == batch_items(recipe, 3, 2, 8)
    @test family_batch_items(stored, :center, 2, 1, 4) ==
          family_batch_items(recipe, :center, 2, 1, 4)
    @test evaluation_batch_items(stored, 1, 5; region=:saddle) ==
          [example_at(recipe, index) for index in (1, 7, 13, 19, 25)]

    # Existing complete splits are verified and reused rather than rewritten.
    reused = materialize_split(recipe, split_root)
    @test dataset_id(reused) == dataset_id(recipe)

    # A copied or damaged bundle must fail checksum verification.
    open(joinpath(split_root, "region.u8"), "r+") do io
      seekstart(io)
      write(io, UInt8(0xff))
    end
    @test_throws ErrorException load_materialized_split(split_root, recipe)
  end
end
