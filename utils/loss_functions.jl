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
    ode_cpu = Float32.(vec(alpha_matrix_key))
    bc_cpu = Float32[series_coeffs[1], series_coeffs[2]]
    data_cpu = Float32.(collect(series_coeffs))

    ode_flat_dev = to_device_fn(ode_cpu)
    bc_dev = to_device_fn(bc_cpu)
    data_dev = to_device_fn(data_cpu)

    # --- W matrix (currently in generate_loss_pde_value Zygote.ignore block) ---
    M = length(ode_cpu)
    W_cpu = zeros(Float32, P, N1)
    for j in 1:P
      for i in 1:N1
        for k in 0:min(i - 1, M - 1)
          W_cpu[j, i] += ode_cpu[k + 1] * xs_cpu[j]^(i - 1 - k) * INV_FACT[i-k]
        end
      end
    end
    W_dev = to_device_fn(W_cpu)

    # --- Power vectors (currently in generate_loss_bc_value Zygote.ignore block) ---
    pow_u_cpu = Float32[x0^(i - 1) * INV_FACT[i] for i in 1:N1]
    pow_du_cpu = Float32[i == 1 ? 0.0f0 : x0^(i - 2) * INV_FACT[i-1] for i in 1:N1]
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
            W_cpu[j, i] += ode_cpu[k + 1] * xs_cpu[j]^(i - 1 - k) * INV_FACT[i-k]
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
      # Fallback: rebuild (used by evaluate_solution / CPU path)
      pow_u_cpu = Float32[x0^(i - 1) * INV_FACT[i] for i in 1:N1]
      pow_du_cpu = Float32[i == 1 ? 0.0f0 : x0^(i - 2) * INV_FACT[i-1] for i in 1:N1]

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

  loss_bc = abs(u_val - bc1) + abs(du_val - bc2)
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
      # Fallback: rebuild (used by evaluate_solution / CPU path)
      d_cpu = Float32.(collect(settings.data))
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

# Keep scalar versions for evaluation/plotting (CPU only, not used in training gradient path)
function ode_residual(settings::LossFunctionSettings, x)
  return sum(
    settings.ode_matrix_flat[order+1] * (
      order == 0 ?
      sum(settings.a_vec[i] * x^(i - 1) / fact[i] for i in 1:settings.n_terms_for_power_series+1) :
      sum(
        settings.a_vec[i] * x^(i - 1 - order) / fact[i-order]
        for i in (order+1):(settings.n_terms_for_power_series+1)
      )
    )
    for order in 0:(length(settings.ode_matrix_flat)-1)
  )
end

function generate_u_approx(settings::LossFunctionSettings)
  u_approx(x) = sum(settings.a_vec[i] * x^(i - 1) / fact[i] for i in 1:(settings.n_terms_for_power_series+1))
  return u_approx
end

export LossFunctionSettings, ODEBuffers, precompute_buffers, generate_loss_pde_value, generate_loss_bc_value, generate_loss_supervised_value, ode_residual, generate_u_approx

end
