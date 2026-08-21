# =============================================================================
# Test: Semigroup Generalization (the "wild ideas" experiment)
# =============================================================================
# The model maps ODE coefficients α to the MONOMIAL coefficients ψ_n of the
# power-series solution u(x) = Σ ψ_n xⁿ, with y(0)=1 (so ψ_n = rⁿ/n! for the
# exponential family).
#
# Held-out trio (α₀·y + α₁·y' = 0 → y = e^{-(α₀/α₁)x}):
#   A = [2; 1]   → e^{-2x}
#   B = [4; 3]   → e^{-(4/3)x}
#   C = [10; 3]  → e^{-(10/3)x} = A·B     (exponents add)
#
# The semigroup statement (internal to the model — no ground truth in it):
#   f_Ω(x; α_A) · f_Ω(x; α_B)  ≈?  f_Ω(x; α_C)
# In monomial basis the product is the Cauchy convolution:
#   c_n = Σ_k  ψ^A_k · ψ^B_{n-k}
#
# Measurements (G-numbers):
#   G_A, G_B, G_C — accuracy on each held-out equation vs. the true e^{rx}
#   G_AB          — the semigroup consistency, model outputs only
# Each reported as max |Δcoefficient| and max function error on [0,1].
#
# Usage:
#   julia --project test/test_semigroup_generalization.jl [path/to/model.checkpoint]
# Default: newest results/run-*/model.checkpoint
# =============================================================================

using Test
using JSON

include("../utils/snapshot_utils.jl")
using .SnapshotUtils

# --- The held-out trio (must match init_batches in src/main.jl) -------------
const A = [2; 1;;]
const B = [4; 3;;]
const C = [10; 3;;]

rate(m) = -m[1, 1] / m[2, 1]                      # y = e^{r·x}, r = -α₀/α₁
canon(m) = Float32.(vec(m) ./ vec(m)[end])        # same convention as canonicalize_alpha

# --- Locate the trained model ------------------------------------------------
function newest_model()
  for ext in (".checkpoint", ".safetensors")
    candidates = String[]
    for run_dir in filter(d -> startswith(d, "run-"), readdir("results"; join=false))
      p = joinpath("results", run_dir, "model$ext")
      isfile(p) && push!(candidates, p)
    end
    isempty(candidates) || return sort(candidates; by=mtime)[end]
  end
  error("No results/run-*/model.checkpoint found — train first (julia --project src/main.jl)")
end

model_path = isempty(ARGS) ? newest_model() : ARGS[1]
@info "Using model: $model_path"

# --- Inference on the trio (network outputs monomial coefficients ψ) ---------
a = Float64.(load_and_infer(model_path, Float32.(A)))
b = Float64.(load_and_infer(model_path, Float32.(B)))
c = Float64.(load_and_infer(model_path, Float32.(C)))
N1 = length(a)   # number of coefficients (series degree N = N1 - 1)

# --- Ground truth: e^{rx} with y(0)=1 → ψ_n = rⁿ/n! ---------------------------
truth(m) = Float64[rate(m)^n / factorial(big(n)) for n in 0:N1-1]

# --- Cauchy convolution of two monomial coefficient vectors (truncated) ------
function cauchy_product(u::Vector{Float64}, v::Vector{Float64})
  n1 = length(u)
  w = zeros(Float64, n1)
  for n in 0:n1-1
    for k in 0:n
      w[n+1] += u[k+1] * v[n-k+1]
    end
  end
  return w
end

# --- Function-space evaluation: u(x) = Σ ψₙ xⁿ on [0,1] -----------------------
function series_eval(d::Vector{Float64}, x::Float64)
  s, xn = 0.0, 1.0
  for n in 0:length(d)-1
    n > 0 && (xn *= x)
    s += d[n+1] * xn
  end
  return s
end
max_fn_err(d1, d2) = maximum(abs(series_eval(d1, x) - series_eval(d2, x)) for x in 0.0:0.01:1.0)
max_coeff_err(d1, d2) = maximum(abs.(d1 .- d2))

# --- The G-numbers ------------------------------------------------------------
ab = cauchy_product(a, b)

results = [
  ("G_A  (A vs e^{-2x})",        a,  truth(A)),
  ("G_B  (B vs e^{-4/3 x})",     b,  truth(B)),
  ("G_C  (C vs e^{-10/3 x})",    c,  truth(C)),
  ("G_AB (A·B vs C, internal)",  ab, c),
]

println("\n", "="^72)
println("Semigroup generalization test — model: $model_path")
println("="^72)
println(rpad("measurement", 32), rpad("max |Δcoeff|", 20), "max fn err on [0,1]")
for (name, d1, d2) in results
  println(rpad(name, 32), rpad(string(round(max_coeff_err(d1, d2), sigdigits=4)), 20),
          round(max_fn_err(d1, d2), sigdigits=4))
end
println("="^72, "\n")

@testset "Semigroup generalization" begin
  # Held-out guarantee: no training matrix may be a scalar multiple of the trio
  @testset "trio (and scalar multiples) excluded from training data" begin
    training_data = JSON.parsefile("data/training_dataset.json")
    trio_canon = Set([canon(A), canon(B), canon(C)])
    for inner in values(training_data), key in keys(inner)
      m = eval(Meta.parse(key))
      @test canon(m) ∉ trio_canon
    end
  end

  @testset "outputs are finite" begin
    @test all(isfinite, a) && all(isfinite, b) && all(isfinite, c) && all(isfinite, ab)
  end
end
