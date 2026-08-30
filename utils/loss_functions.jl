module loss_functions
using CSV
using DataFrames
using Zygote: Zygote

include("./helper_funcs.jl")
using .helper_funcs

# Parametric struct: V can be Vector{Float32} (CPU) or CuVector{Float32} (GPU)
struct LossFunctionSettings{V<:AbstractVector{Float32}}
  a_vec::V
  n_terms_for_power_series::Int
  ode_matrix_flat::V
  x_left::Float32
  boundary_condition::V
  xs::V
  num_points::Int
  num_supervised::Int
  data::V
end

# Pre-computed per-ODE constant data, device-resident (GPU or CPU).
# Created once before the optimization loop and reused as buffers.
struct ODEBuffers{V<:AbstractVector{Float32}, M<:AbstractMatrix{Float32}}
  ode_flat_dev::V      # vec(alpha_matrix_key) on device
  bc_dev::V            # [series_coeffs[1], series_coeffs[2]] on device
  data_dev::V          # series_coeffs on device
  xs_dev::V            # collect(settings.xs) on device (shared ref)
  W::M                 # PDE operator matrix (P × N1)
  pow_u::V             # BC weights for u(x0)
  pow_du::V            # BC derivative weights for Du(x0)
  bc1::Float32         # boundary_condition[1] scalar
  bc2::Float32         # boundary_condition[2] scalar
  padded_data::V       # zero-padded supervised targets (N1,)
  mask::V              # supervised loss mask (N1,)
end

# Pre-compute inverse factorials in Float32 for GPU-compatible loss computation
# 21! fits in Float64; 1/21! ≈ 1.95e-20 is above Float32 minimum
const INV_FACT = Float32.(1.0 ./ factorial.(big.(0:40)))

# MONOMIAL BASIS: the network outputs ψ_n, the coefficients of u(x) = Σ ψ_n xⁿ.
# The dataset stores derivative-basis coefficients a_n = u⁽ⁿ⁾(0); they are
# converted via ψ_n = a_n / n! at load time. ψ_n stays Float32-safe at large N
# (|ψ_n| = |r|ⁿ/n! is bounded) where a_n = rⁿ overflows Float32 when squared.
monomial_from_derivative(series) = Float32[series[i] / factorial(big(i - 1)) for i in 1:length(series)]

# d^k/dx^k xᵐ = m·(m-1)⋯(m-k+1)·x^{m-k} — the falling factorial (1 for k = 0)
falling_factorial(m::Int, k::Int) = k == 0 ? 1.0f0 : Float32(prod(m - t for t in 0:k-1))

"""
    precompute_buffers(settings::PINNSettings, use_gpu::Bool, to_device_fn) → Dict{Any, ODEBuffers}

Pre-compute all constant arrays for every ODE in the training set.
Called once before the optimization loop. `to_device_fn(x)` transfers an array to the target device.
"""
function precompute_buffers(settings, use_gpu::Bool, to_device_fn)
  N1 = settings.n_terms_for_power_series + 1
  x0 = Float32(settings.x_left)
  xs_cpu = Float32.(collect(settings.xs))
  P = length(xs_cpu)
  xs_dev = to_device_fn(xs_cpu)

  buffers = Dict{Any, ODEBuffers}()

  for (alpha_matrix_key, series_coeffs) in settings.ode_matrices
    # --- Device-resident input arrays (currently in loss_fn Zygote.ignore block) ---
    # Canonicalized: scalar multiples of the same ODE become one network input,
    # and the W residual no longer scales with the raw coefficient magnitude.
    ode_cpu = canonicalize_alpha(vec(alpha_matrix_key))
    # BC targets u(0)=a₀=ψ₀ and u'(0)=a₁=ψ₁ are identical in both bases
    bc_cpu = Float32[series_coeffs[1], series_coeffs[2]]
    # Supervised targets in monomial basis: ψ_n = a_n / n!
    data_cpu = monomial_from_derivative(collect(series_coeffs))

    ode_flat_dev = to_device_fn(ode_cpu)
    bc_dev = to_device_fn(bc_cpu)
    data_dev = to_device_fn(data_cpu)

    # --- W matrix (currently in generate_loss_pde_value Zygote.ignore block) ---
    M = length(ode_cpu)
    W_cpu = zeros(Float32, P, N1)
    for j in 1:P
      for i in 1:N1
        for k in 0:min(i - 1, M - 1)
          # monomial basis: coefficient of ψ_{i-1} in u⁽ᵏ⁾ is the falling factorial
          W_cpu[j, i] += ode_cpu[k + 1] * xs_cpu[j]^(i - 1 - k) * falling_factorial(i - 1, k)
        end
      end
    end
    W_dev = to_device_fn(W_cpu)

    # --- Power vectors (currently in generate_loss_bc_value Zygote.ignore block) ---
    pow_u_cpu = Float32[x0^(i - 1) for i in 1:N1]
    pow_du_cpu = Float32[i == 1 ? 0.0f0 : (i - 1) * x0^(i - 2) for i in 1:N1]
    pow_u_dev = to_device_fn(pow_u_cpu)
    pow_du_dev = to_device_fn(pow_du_cpu)
    bc1 = bc_cpu[1]
    bc2 = bc_cpu[2]

    # --- Padded data + mask (currently in generate_loss_supervised_value Zygote.ignore block) ---
    K = settings.num_supervised
    pd_cpu = zeros(Float32, N1)
    m_cpu = zeros(Float32, N1)
    pd_cpu[1:min(K, length(data_cpu))] .= data_cpu[1:min(K, length(data_cpu))]
    m_cpu[1:K] .= 1.0f0
    padded_data_dev = to_device_fn(pd_cpu)
    mask_dev = to_device_fn(m_cpu)

    buffers[alpha_matrix_key] = ODEBuffers(
      ode_flat_dev, bc_dev, data_dev, xs_dev,
      W_dev, pow_u_dev, pow_du_dev, bc1, bc2,
      padded_data_dev, mask_dev
    )
  end

  return buffers
end

# Keep BigFloat factorials for evaluation/plotting code (CPU only)
fact = factorial.(big.(0:21))

"""
    generate_loss_pde_value(settings; ode_buffers=nothing)

GPU-compatible PDE loss using vectorized matrix operations.
When `ode_buffers` is provided, uses the pre-computed W matrix directly.
Otherwise falls back to rebuilding W inside Zygote.ignore().
"""
function generate_loss_pde_value(settings::LossFunctionSettings; ode_buffers::Union{ODEBuffers, Nothing}=nothing)
  N1 = settings.n_terms_for_power_series + 1

  # W is constant w.r.t. a_vec — keep entirely off the AD tape
  W = Zygote.ignore() do
    if ode_buffers !== nothing
      ode_buffers.W
    else
      # Fallback: build W matrix (used by evaluate_solution / CPU path)
      xs_cpu = Float32.(collect(settings.xs))
      ode_cpu = Float32.(collect(settings.ode_matrix_flat))
      P = length(xs_cpu)
      M = length(ode_cpu)

      W_cpu = zeros(Float32, P, N1)
      for j in 1:P
        for i in 1:N1
          for k in 0:min(i - 1, M - 1)
            # monomial basis — must match precompute_buffers
            W_cpu[j, i] += ode_cpu[k + 1] * xs_cpu[j]^(i - 1 - k) * falling_factorial(i - 1, k)
          end
        end
      end

      W_dev = similar(settings.a_vec, P, N1)
      copyto!(W_dev, W_cpu)
      W_dev
    end
  end

  # Single differentiable operation — Zygote handles this on both CPU and GPU
  residual = W * settings.a_vec
  return sum(abs2, residual) / settings.num_points
end

"""
    generate_loss_bc_value(settings; ode_buffers=nothing)

GPU-compatible boundary condition loss.
When `ode_buffers` is provided, uses pre-computed power vectors and BC scalars.
Otherwise falls back to rebuilding inside Zygote.ignore().
"""
function generate_loss_bc_value(settings::LossFunctionSettings; ode_buffers::Union{ODEBuffers, Nothing}=nothing)
  N1 = settings.n_terms_for_power_series + 1
  x0 = settings.x_left

  # Power vectors and BC scalars are constant w.r.t. a_vec — keep off AD tape
  pow_u, pow_du_full, bc1, bc2 = Zygote.ignore() do
    if ode_buffers !== nothing
      (ode_buffers.pow_u, ode_buffers.pow_du, ode_buffers.bc1, ode_buffers.bc2)
    else
      # Fallback: rebuild (used by evaluate_solution / CPU path) — monomial basis
      pow_u_cpu = Float32[x0^(i - 1) for i in 1:N1]
      pow_du_cpu = Float32[i == 1 ? 0.0f0 : (i - 1) * x0^(i - 2) for i in 1:N1]

      pu = similar(settings.a_vec)
      copyto!(pu, pow_u_cpu)
      pdu = similar(settings.a_vec)
      copyto!(pdu, pow_du_cpu)

      bc_cpu = Float32.(collect(settings.boundary_condition))
      (pu, pdu, bc_cpu[1], bc_cpu[2])
    end
  end

  a = settings.a_vec
  u_val = sum(a .* pow_u)
  du_val = sum(a .* pow_du_full)

  # L2 (squared) BC penalty. Both representations and every ported experiment
  # chart use this norm so their loss components are directly comparable.
  loss_bc = abs2(u_val - bc1) + abs2(du_val - bc2)
  return loss_bc
end

"""
    generate_loss_supervised_value(settings; ode_buffers=nothing)

GPU-compatible supervised loss.
When `ode_buffers` is provided, uses pre-computed padded data and mask.
Otherwise falls back to rebuilding inside Zygote.ignore().
"""
function generate_loss_supervised_value(settings::LossFunctionSettings; ode_buffers::Union{ODEBuffers, Nothing}=nothing)
  K = settings.num_supervised
  N1 = length(settings.a_vec)

  # Padded data and mask are constant w.r.t. a_vec — keep off AD tape
  padded_data, mask = Zygote.ignore() do
    if ode_buffers !== nothing
      (ode_buffers.padded_data, ode_buffers.mask)
    else
      # Fallback: rebuild (used by evaluate_solution / CPU path).
      # settings.data carries RAW derivative-basis series — convert to ψ here,
      # mirroring the conversion precompute_buffers applies to padded_data.
      d_cpu = monomial_from_derivative(collect(settings.data))
      pd_cpu = zeros(Float32, N1)
      m_cpu = zeros(Float32, N1)
      pd_cpu[1:min(K, length(d_cpu))] .= d_cpu[1:min(K, length(d_cpu))]
      m_cpu[1:K] .= 1.0f0

      pd = similar(settings.a_vec)
      copyto!(pd, pd_cpu)
      m = similar(settings.a_vec)
      copyto!(m, m_cpu)
      (pd, m)
    end
  end

  diff = (settings.a_vec - padded_data) .* mask
  return sum(abs2, diff) / K
end

# ===========================================================================
# BATCHED POWER-SERIES PATH
# ===========================================================================
#
# The per-ODE path above bakes each ODE's coefficients into its own W matrix,
# so a bin of n ODEs costs n network calls and n matvecs per iteration. That is
# untenable for the transfer experiments (6 models x 600 ODEs x 200k iters).
#
# The fix is to factor the operator the other way. Instead of
#
#     W_b = Σ_k α_k[b] · D_k        (one dense W per ODE b)
#
# keep the D_k SHARED and let the coefficients enter as scalars at residual time:
#
#     resid[:, b] = Σ_k α_k[b] · (D_k · A[:, b])
#
# where D_k[j, i] = d^k/dx^k x^{i-1} evaluated at x_j — the monomial derivative
# matrix for order k, identical for every ODE. Now a whole bin is one network
# call producing A (N1 × nb) and M matmuls of (P × N1)·(N1 × nb).
#
# This is mathematically identical to the per-ODE form (same falling-factorial
# rule, same canonicalized α) and it also uses far less memory: M shared
# matrices instead of one P×N1 matrix per ODE.

# Device-resident constants for a whole bin of ODEs under the power-series
# representation. Built once per bin; the per-order D matrices are shared.
struct BatchBuffers{V<:AbstractVector{Float32}, M<:AbstractMatrix{Float32}}
  X::M            # (in_width × nb) network inputs — canonicalized α, one column per ODE
  ALPHA::M        # (M_orders × nb) canonicalized α coefficients per ODE
  D::Vector{M}    # M_orders matrices, each (P × N1): order-k monomial derivatives
  pow_u::V        # (N1,) evaluates u(x_left)
  pow_du::V       # (N1,) evaluates u'(x_left)
  BC::M           # (2 × nb) [a₀; a₁] targets per ODE
  DATA::M         # (N1 × nb) supervised targets ψ, zero-padded
  MASK::M         # (N1 × nb) 1 on supervised entries, 0 elsewhere
  num_points::Int # P — collocation point count
  num_supervised::Int
  nb::Int         # ODEs in this bin
end

"""
    precompute_batch_buffers(settings, items, use_gpu, to_device_fn) → BatchBuffers

Assemble the batched constants for one bin of ODEs. `items` is a vector of
`alpha_matrix_key => series_coeffs` pairs (a bin from `EpochBatchIterator`).

Everything here is constant w.r.t. the network parameters, so it is built
outside the AD tape and reused every iteration the bin is active.
"""
function precompute_batch_buffers(settings, items, use_gpu::Bool, to_device_fn)
  N1 = settings.n_terms_for_power_series + 1
  x0 = Float32(settings.x_left)
  xs_cpu = Float32.(collect(settings.xs))
  P = length(xs_cpu)
  nb = length(items)
  K = settings.num_supervised

  # Canonicalize every ODE up front so both the network input and the residual
  # coefficients are scale-invariant, exactly as the per-ODE path does.
  alphas = [canonicalize_alpha(vec(k)) for (k, _) in items]
  M_orders = maximum(length, alphas)
  input_width = if settings.input_encoding === :trace_determinant
    2
  elseif !isempty(settings.ode_matrices)
    maximum(prod(size(key)) for (key, _) in settings.ode_matrices)
  else
    M_orders
  end
  if settings.input_encoding !== :trace_determinant
    M_orders <= input_width || error(
      "evaluation ODE order exceeds the trained power-series input width: got $M_orders coefficients, network expects $input_width"
    )
  end

  X_cpu = zeros(Float32, input_width, nb)
  ALPHA_cpu = zeros(Float32, M_orders, nb)
  BC_cpu = zeros(Float32, 2, nb)
  DATA_cpu = zeros(Float32, N1, nb)
  MASK_cpu = zeros(Float32, N1, nb)

  for (b, (_, series)) in enumerate(items)
    a = alphas[b]
    if settings.input_encoding === :trace_determinant
      length(a) == 3 || error("trace-determinant input requires a second-order constant-coefficient ODE")
      X_cpu[:, b] .= Float32[-a[2], a[1]]
    else
      X_cpu[1:length(a), b] .= a
    end
    ALPHA_cpu[1:length(a), b] .= a
    BC_cpu[1, b] = Float32(series[1])
    BC_cpu[2, b] = Float32(series[2])
    # Supervised targets in the monomial basis: ψ_n = a_n / n!
    psi = monomial_from_derivative(collect(series))
    ncopy = min(K, length(psi), N1)
    DATA_cpu[1:ncopy, b] .= psi[1:ncopy]
    MASK_cpu[1:min(K, N1), b] .= 1.0f0
  end

  # Shared order-k monomial derivative matrices:
  #   D_k[j, i] = falling_factorial(i-1, k) * x_j^{i-1-k}
  D_dev = Vector{typeof(to_device_fn(zeros(Float32, P, N1)))}()
  for k in 0:(M_orders - 1)
    Dk = zeros(Float32, P, N1)
    for j in 1:P, i in 1:N1
      if (i - 1) >= k
        Dk[j, i] = falling_factorial(i - 1, k) * xs_cpu[j]^(i - 1 - k)
      end
    end
    push!(D_dev, to_device_fn(Dk))
  end

  pow_u_cpu = Float32[x0^(i - 1) for i in 1:N1]
  pow_du_cpu = Float32[i == 1 ? 0.0f0 : (i - 1) * x0^(i - 2) for i in 1:N1]

  return BatchBuffers(
    to_device_fn(X_cpu), to_device_fn(ALPHA_cpu), D_dev,
    to_device_fn(pow_u_cpu), to_device_fn(pow_du_cpu),
    to_device_fn(BC_cpu), to_device_fn(DATA_cpu), to_device_fn(MASK_cpu),
    P, K, nb,
  )
end

"""
    select_bin(buf::BatchBuffers, idx) → BatchBuffers

A mini-batch view of `buf` containing only the ODEs at column indices `idx`.

Build the buffers ONCE for the whole dataset, then slice per iteration. The
shared per-order derivative matrices, and the BC power vectors, are passed
through by reference — they do not depend on which ODEs are in the bin. Only
the per-ODE columns are sliced, which is cheap and stays on-device.

Rebuilding buffers every iteration instead would re-derive the D matrices each
step and wipe out the entire benefit of batching.
"""
function select_bin(buf::BatchBuffers, idx)
  return BatchBuffers(
    buf.X[:, idx], buf.ALPHA[:, idx], buf.D,
    buf.pow_u, buf.pow_du,
    buf.BC[:, idx], buf.DATA[:, idx], buf.MASK[:, idx],
    buf.num_points, buf.num_supervised, length(idx),
  )
end

"""
    batched_power_series_losses(A, buf::BatchBuffers) → (pde, bc, sup)

The three loss components for a whole bin at once. `A` is the network output,
`(N1 × nb)` — one column of monomial coefficients ψ per ODE.

Every operation is a matmul or a broadcast over full arrays: no scalar
indexing, so this differentiates correctly under Zygote on CPU and GPU alike.
"""
function batched_power_series_losses(A::AbstractMatrix, buf::BatchBuffers)
  # PDE residual: Σ_k α_k ⊙ (D_k · A), summed over derivative orders.
  resid = sum(
    (buf.D[k] * A) .* transpose(view(buf.ALPHA, k, :))
    for k in 1:length(buf.D)
  )
  loss_pde = sum(abs2, resid) / (buf.num_points * buf.nb)

  # BC: u(x₀) and u'(x₀) as 1×nb row vectors, L2 against the targets.
  u_row = transpose(buf.pow_u) * A
  du_row = transpose(buf.pow_du) * A
  loss_bc = (sum(abs2, u_row .- transpose(view(buf.BC, 1, :))) +
             sum(abs2, du_row .- transpose(view(buf.BC, 2, :)))) / buf.nb

  # Supervised: masked MSE against the ψ targets.
  loss_sup = sum(abs2, (A .- buf.DATA) .* buf.MASK) / (buf.num_supervised * buf.nb)

  return (loss_pde, loss_bc, loss_sup)
end

# ===========================================================================
# EIGENVALUE REPRESENTATION
# ===========================================================================
#
# The network outputs four unified-form parameters instead of power-series
# coefficients:
#
#     net : (τ, Δ) → (μ, k, A, B)
#     u(x) = e^{μx} [ A·C(k,x) + B·S(k,x) ]
#
# where C(k,x) = cosh(√k x) and S(k,x) = sinh(√k x)/√k.
#
# Why this form rather than u = c₁e^{λ₁x} + c₂e^{λ₂x}: substituting u = e^{μx}v
# with μ = τ/2 turns y'' - τy' + Δy = 0 into v'' = kv with k = τ²/4 - Δ (the
# discriminant over 4). C and S are ENTIRE functions of k, so:
#
#     k > 0 (real roots)     → cosh/sinh — exponential: saddle, stable/unstable node
#     k < 0 (complex roots)  → cos/sin   — oscillatory: spirals, center
#     k = 0                  → smooth; no branch, no special case
#
# The two-real-exponential ansatz covers the 3 real-root regions. This unified
# form extends to all 6 trace-determinant regions with a single real-valued,
# everywhere-differentiable expression.
#
# The PDE residual is derived analytically from the ansatz rather than
# constructed through a differentiation matrix:
#
#     residual = e^{μx} [ (μ² + k - τμ + Δ)·v + (2μ - τ)·v' ]
#     v = A·C + B·S,   v' = A·k·S + B·C
#
# Minimizing it drives μ → τ/2 and k → τ²/4 - Δ.

# Number of terms in the C/S power series. Pterm = 14 reaches x^29, exact to
# machine precision for |k|x² ≲ 6, which covers the (τ,Δ) box we sample.
const PTERM = 14

# Series coefficients: C = Σ k^n x^{2n}/(2n)!,  S = Σ k^n x^{2n+1}/(2n+1)!
const CS_COEFF_C = Float32[1 / factorial(big(2n)) for n in 0:PTERM]
const CS_COEFF_S = Float32[1 / factorial(big(2n + 1)) for n in 0:PTERM]

# Device-resident constants for one ODE under the eigenvalue representation.
struct EigBuffers{V<:AbstractVector{Float32}, M<:AbstractMatrix{Float32}}
  input_dev::V     # [τ, Δ] — the network input
  XE::M            # (P × PTERM+1) even powers  x^{2n}
  XO::M            # (P × PTERM+1) odd  powers  x^{2n+1}
  cC::V            # (PTERM+1,) 1/(2n)!
  cS::V            # (PTERM+1,) 1/(2n+1)!
  kpow_exp::V      # (PTERM+1,) the exponents 0…PTERM as Float32
  tau::Float32     # trace
  delta::Float32   # determinant
  a0::Float32      # u(0)
  a1::Float32      # u'(0)
  u_true::V        # (P,) analytic solution at the collocation points
end

"""
    tau_delta_from_alpha(alpha) → (τ, Δ)

Convert a flattened ODE coefficient vector `[α₀, α₁, α₂]` representing
`α₂u'' + α₁u' + α₀u = 0` into the trace/determinant form `y'' - τy' + Δy = 0`
used by the eigenvalue representation, i.e. τ = -α₁/α₂ and Δ = α₀/α₂.

Errors on anything that is not 2nd order — the unified ansatz is defined only
for 2nd-order linear ODEs, and truncating a higher-order problem would give
misleading results.
"""
function tau_delta_from_alpha(alpha::AbstractVector)
  a = Float32.(collect(alpha))
  length(a) == 3 || error(
    "eigenvalue representation requires a 2nd-order ODE (3 coefficients [α₀,α₁,α₂]), " *
    "got $(length(a)). Use --representation power_series for this dataset."
  )
  abs(a[3]) > 1f-8 || error(
    "eigenvalue representation requires a nonzero u'' coefficient α₂, got $(a[3])."
  )
  return (-a[2] / a[3], a[1] / a[3])
end

# Analytic solution of y'' - τy' + Δy = 0 with u(0)=a0, u'(0)=a1, evaluated in
# Float64 and narrowed — the closed forms below branch on sign(k), which is fine
# here because this is ground truth and never differentiated through.
function eig_true_solution(tau::Float32, delta::Float32, a0::Float32, a1::Float32, xpts)
  mu = Float64(tau) / 2
  k = Float64(tau)^2 / 4 - Float64(delta)
  A = Float64(a0)
  B = Float64(a1) - mu * Float64(a0)          # u(0)=A, u'(0)=μA+B
  Cf(x) = k >= 0 ? cosh(sqrt(k) * x) : cos(sqrt(-k) * x)
  Sf(x) = abs(k) < 1e-12 ? x :
          (k > 0 ? sinh(sqrt(k) * x) / sqrt(k) : sin(sqrt(-k) * x) / sqrt(-k))
  return Float32[exp(mu * Float64(x)) * (A * Cf(Float64(x)) + B * Sf(Float64(x))) for x in xpts]
end

"""
    precompute_eig_buffers(settings, use_gpu, to_device_fn) → Dict{Any, EigBuffers}

Eigenvalue-representation counterpart of `precompute_buffers`. Builds the
constant per-ODE arrays once, before the optimization loop.
"""
function precompute_eig_buffers(settings, use_gpu::Bool, to_device_fn)
  xs_cpu = Float32.(collect(settings.xs))
  P = length(xs_cpu)

  # Even/odd power matrices are shared by every ODE — build once.
  XE_cpu = Float32[xs_cpu[m]^(2n)     for m in 1:P, n in 0:PTERM]
  XO_cpu = Float32[xs_cpu[m]^(2n + 1) for m in 1:P, n in 0:PTERM]
  XE = to_device_fn(XE_cpu)
  XO = to_device_fn(XO_cpu)
  cC = to_device_fn(CS_COEFF_C)
  cS = to_device_fn(CS_COEFF_S)
  kpow_exp = to_device_fn(Float32.(collect(0:PTERM)))

  buffers = Dict{Any, EigBuffers}()
  for (alpha_matrix_key, series_coeffs) in settings.ode_matrices
    tau, delta = tau_delta_from_alpha(vec(alpha_matrix_key))
    # ICs are basis-independent: u(0)=a₀ and u'(0)=a₁ in every representation.
    a0 = Float32(series_coeffs[1])
    a1 = Float32(series_coeffs[2])

    buffers[alpha_matrix_key] = EigBuffers(
      to_device_fn(Float32[tau, delta]),
      XE, XO, cC, cS, kpow_exp,
      tau, delta, a0, a1,
      to_device_fn(eig_true_solution(tau, delta, a0, a1, xs_cpu)),
    )
  end
  return buffers
end

# C and S at every collocation point, for a scalar k.
# Vectorized as (P × PTERM+1) * (PTERM+1,) matvecs — one differentiable op each,
# no scalar indexing, so this is safe under Zygote on both CPU and GPU.
function _cs_series(k, buf::EigBuffers)
  kpow = k .^ buf.kpow_exp                 # broadcast over a constant exponent vector
  return (buf.XE * (buf.cC .* kpow), buf.XO * (buf.cS .* kpow))
end

"""
    generate_loss_pde_value_eig(out, buf, num_points)

Analytic ODE residual for the unified eigenvalue form. `out` is the network's
4-vector (μ, k, A, B).
"""
function generate_loss_pde_value_eig(out::AbstractVector, buf::EigBuffers, num_points::Int)
  mu, k, A, B = out[1], out[2], out[3], out[4]
  C, S = _cs_series(k, buf)
  v = A .* C .+ B .* S
  vp = A .* (k .* S) .+ B .* C            # v' = A·k·S + B·C
  E = exp.(mu .* buf.XO[:, 1])            # XO[:,1] is x¹, i.e. the raw grid
  resid = E .* ((mu^2 + k - buf.tau * mu + buf.delta) .* v .+ (2 * mu - buf.tau) .* vp)
  return sum(abs2, resid) / num_points
end

"""
    generate_loss_bc_value_eig(out, buf)

Initial-condition loss. In the unified form u(0) = A and u'(0) = μA + B.
"""
function generate_loss_bc_value_eig(out::AbstractVector, buf::EigBuffers)
  mu, A, B = out[1], out[3], out[4]
  # L2, matching the power-series BC term so the two representations' loss
  # components are directly comparable in the viewer.
  return abs2(A - buf.a0) + abs2(mu * A + B - buf.a1)
end

"""
    generate_loss_supervised_value_eig(out, buf)

Supervised loss in SOLUTION space — MSE of the reconstructed u against the
analytic solution at the collocation points. Note this differs from the
power-series path, which supervises in COEFFICIENT space: there is no
coefficient vector here to compare against.
"""
function generate_loss_supervised_value_eig(out::AbstractVector, buf::EigBuffers)
  u = eig_reconstruct(out, buf)
  return sum(abs2, u .- buf.u_true) / length(buf.u_true)
end

# ---------------------------------------------------------------------------
# Batched eigenvalue path
# ---------------------------------------------------------------------------
#
# Same factorization idea as the batched power-series path: every per-ODE
# quantity becomes a 1×nb row, every basis matrix stays shared. The C/S power
# series batches naturally because k enters only as k^n:
#
#     kpow[n, b] = k[b]^n            (PTERM+1 × nb)
#     C = XE · (cC ⊙ kpow)           (P × nb)
#
# so one bin costs one network call plus two matmuls, independent of nb.

struct BatchEigBuffers{V<:AbstractVector{Float32}, M<:AbstractMatrix{Float32}}
  X::M            # (2 × nb) network inputs [τ; Δ]
  TAU::M          # (1 × nb)
  DELTA::M        # (1 × nb)
  xs::V           # (P,) collocation grid
  XE::M           # (P × PTERM+1) even powers x^{2n}
  XO::M           # (P × PTERM+1) odd powers  x^{2n+1}
  cC::V           # (PTERM+1,) 1/(2n)!
  cS::V           # (PTERM+1,) 1/(2n+1)!
  kpow_exp::V     # (PTERM+1,) exponents 0…PTERM
  A0::M           # (1 × nb) u(0) targets
  A1::M           # (1 × nb) u'(0) targets
  UTRUE::M        # (P × nb) analytic solutions
  num_points::Int
  nb::Int
end

"""
    precompute_batch_eig_buffers(settings, items, use_gpu, to_device_fn) → BatchEigBuffers

Batched counterpart of `precompute_eig_buffers`. Errors on any non-2nd-order
ODE in the bin, via `tau_delta_from_alpha`.
"""
function precompute_batch_eig_buffers(settings, items, use_gpu::Bool, to_device_fn)
  xs_cpu = Float32.(collect(settings.xs))
  P = length(xs_cpu)
  nb = length(items)

  X_cpu = zeros(Float32, 2, nb)
  TAU_cpu = zeros(Float32, 1, nb)
  DELTA_cpu = zeros(Float32, 1, nb)
  A0_cpu = zeros(Float32, 1, nb)
  A1_cpu = zeros(Float32, 1, nb)
  UTRUE_cpu = zeros(Float32, P, nb)

  for (b, (key, series)) in enumerate(items)
    tau, delta = tau_delta_from_alpha(vec(key))
    a0 = Float32(series[1])
    a1 = Float32(series[2])
    X_cpu[1, b] = tau; X_cpu[2, b] = delta
    TAU_cpu[1, b] = tau; DELTA_cpu[1, b] = delta
    A0_cpu[1, b] = a0;  A1_cpu[1, b] = a1
    UTRUE_cpu[:, b] .= eig_true_solution(tau, delta, a0, a1, xs_cpu)
  end

  XE_cpu = Float32[xs_cpu[m]^(2n)     for m in 1:P, n in 0:PTERM]
  XO_cpu = Float32[xs_cpu[m]^(2n + 1) for m in 1:P, n in 0:PTERM]

  return BatchEigBuffers(
    to_device_fn(X_cpu), to_device_fn(TAU_cpu), to_device_fn(DELTA_cpu),
    to_device_fn(xs_cpu), to_device_fn(XE_cpu), to_device_fn(XO_cpu),
    to_device_fn(CS_COEFF_C), to_device_fn(CS_COEFF_S),
    to_device_fn(Float32.(collect(0:PTERM))),
    to_device_fn(A0_cpu), to_device_fn(A1_cpu), to_device_fn(UTRUE_cpu),
    P, nb,
  )
end

"""
    batched_reconstruct(out, buf) → u  (P × nb)

The solution values at the collocation points, for a whole bin.

This is what makes the two representations comparable: coefficient-space error
is not directly comparable across them (N+1 monomial coefficients vs 4
unified-form parameters), but solution-space error is basis-independent.
Every cross-region and sweep metric is computed from this.

For the power-series form `D[1]` is the order-0 derivative matrix, i.e. plain
monomial evaluation x^{i-1}, so `u = D[1] · A` exactly.
"""
batched_reconstruct(A::AbstractMatrix, buf::BatchBuffers) = buf.D[1] * A

function batched_reconstruct(O::AbstractMatrix, buf::BatchEigBuffers)
  mu = view(O, 1:1, :); k = view(O, 2:2, :)
  A = view(O, 3:3, :);  B = view(O, 4:4, :)
  kpow = k .^ reshape(buf.kpow_exp, :, 1)
  C = buf.XE * (reshape(buf.cC, :, 1) .* kpow)
  S = buf.XO * (reshape(buf.cS, :, 1) .* kpow)
  return exp.(buf.xs * mu) .* (A .* C .+ B .* S)
end

"""
    true_solutions(buf) → u_true  (P × nb)

Ground-truth solution values for a bin, so the relative-L2 metric has a
reference. The eigenvalue buffers carry these analytically; the power-series
buffers carry supervised ψ targets, from which the solution is the same
monomial evaluation applied to the targets rather than the prediction.
"""
true_solutions(buf::BatchEigBuffers) = buf.UTRUE
true_solutions(buf::BatchBuffers) = buf.D[1] * buf.DATA

"""
    relative_l2(u_pred, u_true) → Float32

Solution-space relative L2 error over a whole bin — the basis-independent
metric used for every transfer and sweep chart.
"""
relative_l2(u_pred, u_true) =
  Float32(sqrt(sum(abs2, u_pred .- u_true) / (sum(abs2, u_true) + 1f-12)))

"""
    select_bin(buf::BatchEigBuffers, idx) → BatchEigBuffers

Mini-batch view over ODE columns. The shared basis matrices (XE, XO, the C/S
coefficients, the collocation grid) pass through by reference; only the per-ODE
columns are sliced. Notably this avoids recomputing `UTRUE`, the analytic
solutions, which are the costliest part of the eigenvalue buffers.
"""
function select_bin(buf::BatchEigBuffers, idx)
  return BatchEigBuffers(
    buf.X[:, idx], buf.TAU[:, idx], buf.DELTA[:, idx],
    buf.xs, buf.XE, buf.XO, buf.cC, buf.cS, buf.kpow_exp,
    buf.A0[:, idx], buf.A1[:, idx], buf.UTRUE[:, idx],
    buf.num_points, length(idx),
  )
end

"""
    batched_eigenvalue_losses(O, buf::BatchEigBuffers) → (pde, bc, sup)

The three loss components for a whole bin. `O` is the network output,
`(4 × nb)` — rows are (μ, k, A, B).
"""
function batched_eigenvalue_losses(O::AbstractMatrix, buf::BatchEigBuffers)
  mu = view(O, 1:1, :); k = view(O, 2:2, :)
  A = view(O, 3:3, :);  B = view(O, 4:4, :)

  # These arrays define the fixed ODE batch and analytic basis. Gradients are
  # required only with respect to O (and therefore the network parameters).
  # Without this barrier Zygote constructs adjoints for the CuArray constants;
  # some of those adjoints fall back to scalar-indexed generic matvec code.
  tau, delta, xs, xe, xo, cc, cs, exponents, a0, a1, utrue = Zygote.ignore() do
    (buf.TAU, buf.DELTA, buf.xs, buf.XE, buf.XO, buf.cC, buf.cS,
     buf.kpow_exp, buf.A0, buf.A1, buf.UTRUE)
  end

  kpow = k .^ reshape(exponents, :, 1) # (PTERM+1 × nb): kpow[n,b] = k[b]^n
  C = xe * (reshape(cc, :, 1) .* kpow) # (P × nb)
  S = xo * (reshape(cs, :, 1) .* kpow)

  v = A .* C .+ B .* S
  vp = A .* (k .* S) .+ B .* C          # v' = A·k·S + B·C
  E = exp.(xs * mu)                     # (P × nb)

  resid = E .* ((mu .^ 2 .+ k .- tau .* mu .+ delta) .* v .+
                (2 .* mu .- tau) .* vp)
  loss_pde = sum(abs2, resid) / (buf.num_points * buf.nb)

  # u(0) = A, u'(0) = μA + B — L2, matching the power-series BC term.
  loss_bc = (sum(abs2, A .- a0) + sum(abs2, mu .* A .+ B .- a1)) / buf.nb

  loss_sup = sum(abs2, (E .* v) .- utrue) / (buf.num_points * buf.nb)

  return (loss_pde, loss_bc, loss_sup)
end

"""
    eig_reconstruct(out, buf) → u at the collocation points

u(x) = e^{μx}[A·C(k,x) + B·S(k,x)]. Shared by the supervised loss and by
evaluation/plotting so the two can never drift apart.
"""
function eig_reconstruct(out::AbstractVector, buf::EigBuffers)
  mu, k, A, B = out[1], out[2], out[3], out[4]
  C, S = _cs_series(k, buf)
  return exp.(mu .* buf.XO[:, 1]) .* (A .* C .+ B .* S)
end

# Keep scalar versions for evaluation/plotting (CPU only, not used in training gradient path)
# a_vec holds monomial coefficients ψ: u(x) = Σ ψ_{i-1} x^{i-1}
function ode_residual(settings::LossFunctionSettings, x)
  return sum(
    settings.ode_matrix_flat[order+1] * (
      order == 0 ?
      sum(settings.a_vec[i] * x^(i - 1) for i in 1:settings.n_terms_for_power_series+1) :
      sum(
        settings.a_vec[i] * falling_factorial(i - 1, order) * x^(i - 1 - order)
        for i in (order+1):(settings.n_terms_for_power_series+1)
      )
    )
    for order in 0:(length(settings.ode_matrix_flat)-1)
  )
end

function generate_u_approx(settings::LossFunctionSettings)
  u_approx(x) = sum(settings.a_vec[i] * x^(i - 1) for i in 1:(settings.n_terms_for_power_series+1))
  return u_approx
end

export LossFunctionSettings, ODEBuffers, precompute_buffers, generate_loss_pde_value, generate_loss_bc_value, generate_loss_supervised_value, ode_residual, generate_u_approx, monomial_from_derivative, falling_factorial
export BatchBuffers, precompute_batch_buffers, batched_power_series_losses
export BatchEigBuffers, precompute_batch_eig_buffers, batched_eigenvalue_losses
export select_bin, batched_reconstruct, true_solutions, relative_l2
export EigBuffers, precompute_eig_buffers, generate_loss_pde_value_eig, generate_loss_bc_value_eig, generate_loss_supervised_value_eig, eig_reconstruct, eig_true_solution, tau_delta_from_alpha

end
