using Test
using JSON

include("../scripts/shared/bundle_study.jl")

function write_fixture(root, relative, contents="fixture")
  path = joinpath(root, relative)
  mkpath(dirname(path))
  write(path, contents)
  return path
end

@testset "Portable study bundle" begin
  mktempdir() do root
    required = [
      "config.json",
      "config.md",
      "environment/source-manifest.json",
      "environment/source/Project.toml",
      "environment/source/Manifest.toml",
      "datasets/train/manifest.json",
      "datasets/validation/manifest.json",
      "datasets/test/manifest.json",
      "datasets/inference/points.json",
      "selection/validation-metrics.csv",
      "selection/selected-checkpoints.json",
      "data/final-test-metrics.json",
      "data/batch_transfer_power_series.json",
      "data/batch_transfer_eigenvalue.json",
      "data/generalization-suite.json",
      "charts/overview.png",
    ]
    foreach(relative -> write_fixture(root, relative), required)

    representations = ("power_series", "eigenvalue")
    regions = ("saddle", "stable_node", "unstable_node",
               "stable_spiral", "unstable_spiral", "center")
    for representation in representations
      write_fixture(root, "global/$representation/model-final.checkpoint")
      write_fixture(root, "models/selected/global/$representation.checkpoint")
    end
    for region in regions, representation in representations
      write_fixture(root, "family/$region/$representation/model-final.checkpoint")
      write_fixture(root, "models/selected/family/$region/$representation.checkpoint")
    end
    # Periodic snapshots are artifacts, but they are not additional final models.
    write_fixture(root, "global/power_series/snapshots/epoch-0005.checkpoint")

    finalize(root)
    manifest = JSON.parsefile(joinpath(root, "bundle-manifest.json"))
    @test manifest["final_checkpoint_count"] == 14
    @test manifest["selected_checkpoint_count"] == 14
    @test manifest["chart_count"] == 1
    @test verify(root) === nothing

    write_fixture(root, "unexpected.txt")
    @test_throws ErrorException verify(root)
  end
end
