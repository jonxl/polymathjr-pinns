module SnapshotUtils

using ComponentArrays
using Lux
import Random

include("../utils/safetensors_utils.jl")
using .SafeTensorSnapshots

include("../utils/helper_funcs.jl")
using .helper_funcs

"""
    load_and_infer(snapshot_path::String, ode_matrix::Matrix)

Load model from a `.safetensors` file and run inference to get coefficients.
The file is fully self-contained — no external PINNSettings needed.

Returns a vector of predicted coefficients.
"""
function load_and_infer(snapshot_path::String, ode_matrix::Matrix)
  coeff_net, p, st, _ = SafeTensorSnapshots.load_model(snapshot_path)
  matrix_flat = canonicalize_alpha(vec(ode_matrix))  # must match training input convention
  coefficients = first(coeff_net(matrix_flat, p, st))[:, 1]
  return coefficients
end

"""
    construct_solution(ψ::AbstractVector) → f_Ω

Turn the model's output — the monomial coefficient vector ψ — into the callable
constructed function f_Ω(x) = Σ ψ_n xⁿ (evaluated via Horner's method).
"""
function construct_solution(ψ::AbstractVector)
  coeffs = Tuple(Float64.(ψ))   # evalpoly expects ascending-order coefficients
  return x -> evalpoly(Float64(x), coeffs)
end

"""
    load_solution(snapshot_path::String, ode_matrix::Matrix) → f_Ω

One step from a saved model + ODE matrix to the callable solution:
inference (`load_and_infer`) composed with `construct_solution`.
"""
load_solution(snapshot_path::String, ode_matrix::Matrix) =
  construct_solution(load_and_infer(snapshot_path, ode_matrix))

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

export load_and_infer, replay_snapshots, construct_solution, load_solution

end
