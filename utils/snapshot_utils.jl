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
