module SnapshotUtils

using ComponentArrays
using Lux
import Random

# PINN module is provided by the parent (training_schemes) — do not re-include
using ..PINN

include("../utils/safetensors_utils.jl")
using .SafeTensorSnapshots

"""
    load_and_infer(snapshot_path::String, settings::PINNSettings, ode_matrix::Matrix)

Load saved model weights from a `.safetensors` snapshot and run inference to get coefficients.
Legacy raw `.bin` snapshots are still readable.

The network architecture is rebuilt deterministically from `PINNSettings` via `initialize_network`,
so the `ComponentArray` axis layout is always identical to what was saved.

Returns a vector of predicted coefficients.
"""
function load_and_infer(snapshot_path::String, settings::PINNSettings, ode_matrix::Matrix)
  # 1. Rebuild network from settings (deterministic from neuron_num, n_terms, seed)
  coeff_net, p_template, st = initialize_network(settings)

  # 2. Load saved weights into the same ComponentArray layout
  raw = load_snapshot_vector(snapshot_path)
  p = ComponentArray(raw, getaxes(p_template))

  # 3. Run inference
  matrix_flat = Float32.(vec(ode_matrix))
  coefficients = first(coeff_net(matrix_flat, p, st))[:, 1]

  return coefficients
end

"""
    replay_snapshots(run_dir::String, settings::PINNSettings, ode_matrix::Matrix)

Sweep every `.safetensors` snapshot in `run_dir`, run inference on each with the given
ODE matrix, and return how the model's coefficient predictions evolve over training.
Legacy raw `.bin` files are included if present.

Returns a vector of named tuples: `[(iteration=Int, coefficients=Vector{Float32}), ...]`
sorted by iteration number.
"""
function replay_snapshots(run_dir::String, settings::PINNSettings, ode_matrix::Matrix)
  # Find all safetensors snapshots, plus legacy .bin files, and sort by iteration number.
  snapshot_files = filter(f -> endswith(f, ".safetensors") || endswith(f, ".bin"), readdir(run_dir))
  sort!(snapshot_files)

  results = NamedTuple[]
  for fname in snapshot_files
    # Extract iteration from filename: "iter-0001000.safetensors" → 1000
    m = match(r"iter-(\d+)\.(?:safetensors|bin)", fname)
    m === nothing && continue
    iteration = parse(Int, m.captures[1])

    snapshot_path = joinpath(run_dir, fname)
    coefficients = load_and_infer(snapshot_path, settings, ode_matrix)
    push!(results, (iteration=iteration, coefficients=Vector{Float32}(coefficients)))
  end

  return results
end

export load_and_infer, replay_snapshots

end
