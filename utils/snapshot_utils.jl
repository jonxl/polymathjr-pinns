module SnapshotUtils

using ComponentArrays
using Lux
import Random

include("../utils/safetensors_utils.jl")
using .SafeTensorSnapshots

include("../utils/helper_funcs.jl")
using .helper_funcs

include("../utils/loss_functions.jl")
using .loss_functions

"""
    load_and_infer(snapshot_path::String, ode_matrix::Matrix)

Load model from a `.checkpoint` (raw) or `.safetensors` file and run inference to get coefficients.
The file is fully self-contained — no external PINNSettings needed.

Returns a vector of predicted coefficients.
"""
function load_and_infer(snapshot_path::String, ode_matrix::Matrix)
  coeff_net, p, st, metadata = SafeTensorSnapshots.load_any_model(snapshot_path)
  representation = Symbol(get(metadata, "representation", "power_series"))
  input_encoding = Symbol(get(metadata, "input_encoding",
                              representation === :eigenvalue ? "trace_determinant" : "coefficients"))
  input = if input_encoding === :trace_determinant
    tau, delta = tau_delta_from_alpha(vec(ode_matrix))
    Float32[tau, delta]
  else
    canonicalize_alpha(vec(ode_matrix))
  end
  return first(coeff_net(input, p, st))[:, 1]
end

"""
    construct_solution(output::AbstractVector; representation=:power_series) → f_Ω

Turn a model output into the callable constructed function.
"""
function construct_solution(output::AbstractVector; representation::Symbol=:power_series)
  vals = Float64.(output)
  if representation === :eigenvalue
    mu, k, A, B = vals
    C(x) = k >= 0 ? cosh(sqrt(k) * x) : cos(sqrt(-k) * x)
    S(x) = abs(k) < 1e-12 ? x :
           (k > 0 ? sinh(sqrt(k) * x) / sqrt(k) : sin(sqrt(-k) * x) / sqrt(-k))
    return x -> exp(mu * Float64(x)) * (A * C(Float64(x)) + B * S(Float64(x)))
  end
  coeffs = Tuple(vals)
  return x -> evalpoly(Float64(x), coeffs)
end

"""
    load_solution(snapshot_path::String, ode_matrix::Matrix) → f_Ω

One step from a saved model + ODE matrix to the callable solution:
inference (`load_and_infer`) composed with `construct_solution`.
"""
function load_solution(snapshot_path::String, ode_matrix::Matrix)
  _, _, _, metadata = SafeTensorSnapshots.load_any_model(snapshot_path)
  representation = Symbol(get(metadata, "representation", "power_series"))
  return construct_solution(load_and_infer(snapshot_path, ode_matrix);
                            representation=representation)
end

"""
    replay_snapshots(run_dir::String, ode_matrix::Matrix)

Sweep every `.checkpoint`/`.safetensors` snapshot in `run_dir`, run inference on each, and
return how coefficient predictions evolve over training.
No external settings needed — each file is self-contained.

Returns a vector of named tuples: [(iteration=Int, coefficients=Vector{Float32}), ...]
"""
function replay_snapshots(run_dir::String, ode_matrix::Matrix)
  snapshot_files = filter(f -> endswith(f, ".checkpoint") || endswith(f, ".safetensors"), readdir(run_dir))
  sort!(snapshot_files)

  results = NamedTuple[]
  for fname in snapshot_files
    m = match(r"iter-(\d+).*\.(checkpoint|safetensors)$", fname)
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
