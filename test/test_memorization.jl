# =============================================================================
# Test: Memorization — verify the PINN recovers a training-set ODE to high
# precision. The model has seen this ODE during training, so all three loss
# components (BC, PDE, supervised) should be near machine precision.
# =============================================================================
# The test picks the first ODE from batch "01" in data/training_dataset.json,
# loads the trained model, canonicalizes the ODE matrix, and calls loss_fn
# to compute boundary-condition, PDE-residual, and supervised-coefficient
# losses. No magnitude thresholds are enforced — the raw values are printed
# so the user can assess memorization quality.
#
# Usage:
#   julia --project test/test_memorization.jl [path/to/model.safetensors]
# Default: newest results/run-*/model.safetensors
# =============================================================================

using Test
using JSON
using Printf

include("../utils/safetensors_utils.jl")
using .SafeTensorSnapshots

include("../utils/helper_funcs.jl")
using .helper_funcs

include("../architectures/PINN.jl")
using .PINN

# --- Locate the trained model ------------------------------------------------
function newest_model()
  candidates = String[]
  for run_dir in filter(d -> startswith(d, "run-"), readdir("results"; join=false))
    p = joinpath("results", run_dir, "model.safetensors")
    isfile(p) && push!(candidates, p)
  end
  isempty(candidates) && error("No results/run-*/model.safetensors found — train first (julia --project src/main.jl)")
  return sort(candidates; by=mtime)[end]
end

model_path = isempty(ARGS) ? newest_model() : ARGS[1]
@info "Using model: $model_path"

# --- Load one training example from the training dataset ----------------------
training_data = JSON.parsefile("data/training_dataset.json")
@assert !isempty(training_data) "Training dataset is empty — run --gen-data first"

batch_key = first(keys(training_data))       # e.g. "01"
inner = training_data[batch_key]
@assert !isempty(inner) "Training dataset batch \"$batch_key\" is empty"

matrix_key_str, series_coeffs = first(inner)
ode_matrix = eval(Meta.parse(matrix_key_str))   # string → Matrix{Int}
coeffs = collect(Float64, series_coeffs)        # derivative-basis coefficients

n_coeffs = length(coeffs)
N = n_coeffs - 1                                 # n_terms_for_power_series
num_supervised = min(10, n_coeffs)
num_points = 10

# --- Load the trained model ---------------------------------------------------
coeff_net, p, st, _ = SafeTensorSnapshots.load_model(model_path)

# --- Construct a minimal PINNSettings for loss_fn ------------------------------
# loss_fn only reads: n_terms_for_power_series, num_supervised, num_points,
# x_left, xs, pde_weight, supervised_weight, optimizer.
# The other fields are dummy values (never accessed by loss_fn).
dummy_ode_matrices = Dict{Any,Any}()

settings = PINNSettings(
  0,                               # neuron_num       (dummy)
  0,                               # seed             (dummy)
  dummy_ode_matrices,              # ode_matrices     (dummy)
  0,                               # maxiters_lbfgs   (dummy)
  N,                               # n_terms_for_power_series
  num_supervised,                  # num_supervised
  num_points,                      # num_points
  Float32(0.0),                    # x_left
  Float32(1.0),                    # x_right
  Float32(1.0),                    # supervised_weight
  Float32(1.0),                    # pde_weight
  range(Float32(0.0), Float32(1.0), length=num_points),  # xs
  "adam",                          # optimizer
)

# --- Prepare ODE inputs for loss_fn -----------------------------------------
matrix_flat = canonicalize_alpha(vec(ode_matrix))
boundary_condition = Float32[coeffs[1], coeffs[2]]

# --- Compute losses ---------------------------------------------------------
total, bc, pde, sup = PINN.loss_fn(
  p, coeffs, coeff_net, st,
  matrix_flat, boundary_condition,
  settings, false               # use_gpu = false (CPU path)
)

# --- Print results -------------------------------------------------------------
println("\n", "="^72)
println("Memorization test — model: ", model_path)
println("Training ODE: ", matrix_key_str)
println("  Coefficients : ", n_coeffs, " terms (N = ", N, ")")
println("  Supervised   : ", num_supervised)
println("="^72)
@printf("  Loss (total weighted) : %s\n", string(total))
@printf("  Loss (BC)             : %s\n", string(bc))
@printf("  Loss (PDE)            : %s\n", string(pde))
@printf("  Loss (supervised)     : %s\n", string(sup))
println("="^72)
println()

# --- Sanity assertions (no magnitude thresholds) --------------------------------
@testset "memorization" begin
  @test isfile(model_path)
  @test N >= 0
  @test all(isfinite, [total, bc, pde, sup])
end
