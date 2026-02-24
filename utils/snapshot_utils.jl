module SnapshotUtils

using ComponentArrays
using Lux
import Random

# PINN module is provided by the parent (training_schemes) — do not re-include
using ..PINN

"""
    load_and_infer(snapshot_path::String, settings::PINNSettings, ode_matrix::Matrix)

Load saved model weights from a `.bin` snapshot and run inference to get coefficients.

The network architecture is rebuilt deterministically from `PINNSettings` via `initialize_network`,
so the `ComponentArray` axis layout is always identical to what was saved.

Returns a vector of predicted coefficients.
"""
function load_and_infer(snapshot_path::String, settings::PINNSettings, ode_matrix::Matrix)
  # 1. Rebuild network from settings (deterministic from neuron_num, n_terms, seed)
  coeff_net, p_template, st = initialize_network(settings)

  # 2. Load saved weights into the same ComponentArray layout
  raw = reinterpret(Float32, read(snapshot_path))
  p = ComponentArray(raw, getaxes(p_template))

  # 3. Run inference
  matrix_flat = Float32.(vec(ode_matrix))
  coefficients = first(coeff_net(matrix_flat, p, st))[:, 1]

  return coefficients
end

"""
    replay_snapshots(run_dir::String, settings::PINNSettings, ode_matrix::Matrix)

Sweep every `.bin` snapshot in `run_dir`, run inference on each with the given
ODE matrix, and return how the model's coefficient predictions evolve over training.

Returns a vector of named tuples: `[(iteration=Int, coefficients=Vector{Float32}), ...]`
sorted by iteration number.
"""
function replay_snapshots(run_dir::String, settings::PINNSettings, ode_matrix::Matrix)
  # Find all .bin files and sort by iteration number
  bin_files = filter(f -> endswith(f, ".bin"), readdir(run_dir))
  sort!(bin_files)

  results = NamedTuple[]
  for fname in bin_files
    # Extract iteration from filename: "iter-0001000.bin" → 1000
    m = match(r"iter-(\d+)\.bin", fname)
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
