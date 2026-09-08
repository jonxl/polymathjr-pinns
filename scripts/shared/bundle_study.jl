#!/usr/bin/env julia

using Dates
using JSON
using SHA

function usage()
  println("""
  Finalize or verify a portable dual-representation study bundle.

  Usage:
    julia --project=. scripts/shared/bundle_study.jl finalize STUDY_ROOT
    julia --project=. scripts/shared/bundle_study.jl verify   STUDY_ROOT
  """)
end

function sha256_file(path)
  open(path, "r") do io
    return bytes2hex(sha256(io))
  end
end

function required_paths(root)
  paths = [
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
    "models/selected",
    "data/final-test-metrics.json",
    "data/batch_transfer_power_series.json",
    "data/batch_transfer_eigenvalue.json",
    "data/generalization-suite.json",
    "charts",
  ]
  for relative in paths
    ispath(joinpath(root, relative)) || error("Study is incomplete; missing $relative")
  end
end

function artifact_files(root)
  manifest_path = abspath(joinpath(root, "bundle-manifest.json"))
  files = String[]
  for (dir, _, names) in walkdir(root), name in names
    path = abspath(joinpath(dir, name))
    path == manifest_path && continue
    name in (".DS_Store",) && continue
    push!(files, path)
  end
  sort!(files; by=path -> relpath(path, root))
  return files
end

function final_checkpoint_count(root)
  regions = ("saddle", "stable_node", "unstable_node",
             "stable_spiral", "unstable_spiral", "center")
  representations = ("power_series", "eigenvalue")
  dirs = [joinpath(root, "global", representation)
          for representation in representations]
  append!(dirs, [joinpath(root, "family", region, representation)
                 for region in regions for representation in representations])
  count = 0
  for dir in dirs
    isdir(dir) || continue
    finals = [name for name in readdir(dir)
              if startswith(name, "model-") && endswith(name, ".checkpoint")]
    length(finals) == 1 || error("Expected exactly one final checkpoint in $dir")
    count += 1
  end
  return count
end

function selected_checkpoint_count(root)
  dir = joinpath(root, "models", "selected")
  isdir(dir) || return 0
  return count(path -> endswith(path, ".checkpoint"),
    (joinpath(parent, name) for (parent, _, names) in walkdir(dir) for name in names))
end

function finalize(root)
  required_paths(root)
  final_checkpoint_count(root) == 14 || error("Expected exactly 14 final model checkpoints")
  selected_checkpoint_count(root) == 14 || error("Expected exactly 14 validation-selected model checkpoints")
  files = artifact_files(root)
  entries = [Dict(
    "path" => relpath(path, root),
    "bytes" => filesize(path),
    "sha256" => sha256_file(path),
  ) for path in files]
  chart_count = count(entry -> startswith(entry["path"], "charts/") &&
                               endswith(entry["path"], ".png"), entries)
  chart_count > 0 || error("No PNG charts found under $(joinpath(root, "charts"))")
  payload = Dict(
    "format" => "polymathjr-portable-study-bundle",
    "format_version" => 1,
    "status" => "complete",
    "finalized_at" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
    "file_count" => length(entries),
    "total_bytes" => sum(entry["bytes"] for entry in entries),
    "final_checkpoint_count" => 14,
    "selected_checkpoint_count" => 14,
    "chart_count" => chart_count,
    "files" => entries,
  )
  open(joinpath(root, "bundle-manifest.json"), "w") do io
    JSON.print(io, payload, 2)
  end
  println("Finalized portable study bundle: $root")
  println("  files: $(payload["file_count"])")
  println("  bytes: $(payload["total_bytes"])")
  println("  charts: $(payload["chart_count"])")
end

function verify(root)
  path = joinpath(root, "bundle-manifest.json")
  isfile(path) || error("Missing bundle manifest: $path")
  payload = JSON.parsefile(path)
  payload["format"] == "polymathjr-portable-study-bundle" || error("Unsupported bundle format")
  failures = String[]
  listed = Set(String(entry["path"]) for entry in payload["files"])
  actual = Set(relpath(path, root) for path in artifact_files(root))
  for relative in sort!(collect(setdiff(actual, listed)))
    push!(failures, "unexpected: $relative")
  end
  for relative in sort!(collect(setdiff(listed, actual)))
    push!(failures, "missing: $relative")
  end
  for entry in payload["files"]
    artifact = joinpath(root, entry["path"])
    if !isfile(artifact)
      push!(failures, "missing: $(entry["path"])")
      continue
    end
    filesize(artifact) == entry["bytes"] || push!(failures, "size: $(entry["path"])")
    sha256_file(artifact) == entry["sha256"] || push!(failures, "sha256: $(entry["path"])")
  end
  length(payload["files"]) == payload["file_count"] || push!(failures, "file_count")
  sum(entry["bytes"] for entry in payload["files"]) == payload["total_bytes"] ||
    push!(failures, "total_bytes")
  isempty(failures) || error("Bundle verification failed:\n  " * join(failures, "\n  "))
  println("Verified portable study bundle: $root ($(length(payload["files"])) files)")
end

function main(args=ARGS)
  length(args) == 2 || (usage(); return 1)
  mode, root = args
  mode == "finalize" ? finalize(root) :
  mode == "verify" ? verify(root) : (usage(); error("Unknown mode: $mode"))
  return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
  exit(main())
end
