module Plugboard

using LinearAlgebra
using TaylorSeries
using Random
using JSON
using Logging
using ProgressMeter

DEBUG = false # Toggle verbose debug output

#= Enable @debug level logging when DEBUG is true
function __init__()
  if DEBUG
    global_logger(ConsoleLogger(stderr, Logging.Debug))
  end
end
=#

struct Settings
  ode_order::Int
  poly_degree::Int
  dataset_size::Int
  data_dir::String
  num_of_terms::Int
end

# Generate random alpha matrix - unchanged
function generate_random_alpha_matrix(ode_order, poly_degree)
  rows = ode_order + 1
  cols = poly_degree + 1
  α_matrix = Matrix{Int}(undef, rows, cols)
  for i in 1:rows
    for j in 1:cols
      α_matrix[i, j] = rand(Bool) ? rand(-10:-1) : rand(1:10)
    end
  end
  return α_matrix
end

# Generate matrices based on the constraint a^2 - 4b > 0
"""
    generate_random_alpha_matrix_with_constraint(ode_order, poly_degree; coeff_bound, max_ratio)

Random MONIC ODE coefficient matrix: the leading (highest-order) coefficient is
always exactly 1, so an order-m ODE has m free parameters rather than m+1.

    order 1:  y' + α₀y = 0                  α = [α₀; 1]
    order 2:  y'' + α₁y' + α₀y = 0          α = [α₀; α₁; 1]

WHY MONIC. `helper_funcs.canonicalize_alpha` already divides every ODE by its
leading coefficient before the network ever sees it, so [722; -510] and
[-1.4157; 1] are the SAME network input. Drawing a leading coefficient and then
dividing it out produced redundant dataset entries and needed a rejection loop
to control the ratio that actually mattered. Fixing it at 1 removes both: the
free parameter IS the quantity that determines the solution.

This also makes the random generator agree with `alpha_from_tau_delta`, which
emits the monic `[Δ; -τ; 1]` for the trace-determinant family.

Coefficients are REAL, not integer. At order 1 the solution is y = A·e^{-α₀x}
with series coefficients aₙ = (-α₀)ⁿ, so `max_ratio` bounds |α₀| to keep aₙ
inside Float32 range — the same guarantee the old ratio cap provided. Integers
would allow only ~2·max_ratio distinct ODEs, collapsing dataset diversity,
whereas the old rational ratios (e.g. 722/-510 = -1.4157) were fine-grained.

`coeff_bound` is accepted for signature compatibility and no longer used: with
a monic leading coefficient there is no scale left to bound, only the ratio.
"""
function generate_random_alpha_matrix_with_constraint(ode_order, poly_degree; coeff_bound::Int=1000, max_ratio::Int=10)
  rows = ode_order + 1
  cols = poly_degree + 1

  # Draw a coefficient in ±max_ratio, away from zero so no term degenerates.
  lo = 0.05
  draw() = (rand(Bool) ? 1.0 : -1.0) * (lo + rand() * (max_ratio - lo))

  α_matrix = zeros(Float64, rows, cols)

  if cols == 1
    # Constant coefficients: free parameters α₀…α_{m-1}, leading coefficient 1.
    for i in 1:(rows - 1)
      α_matrix[i, 1] = draw()
    end
    α_matrix[rows, 1] = 1.0
  else
    # Variable (polynomial) coefficients: keep the leading term monic and
    # constant in x, so the ODE stays order-m with a nonsingular leading
    # coefficient; the lower-order terms may carry x^j factors.
    for i in 1:(rows - 1), j in 1:cols
      α_matrix[i, j] = draw()
    end
    α_matrix[rows, 1] = 1.0
  end

  return α_matrix
end

# Canonical form of an ODE matrix: divide by the leading (last) coefficient,
# mirroring helper_funcs.canonicalize_alpha. Scalar multiples of the same ODE
# share one canonical key — used for held-out exclusion.
canonical_matrix_key(α_matrix) = string(Float32.(vec(α_matrix) ./ vec(α_matrix)[end]))

# ===========================================================================
# TRACE-DETERMINANT REGION SAMPLING
# ===========================================================================
#
# The generalization experiments do not want uniformly random ODEs — they want
# ODEs drawn from a NAMED REGION of the trace-determinant plane. For
#
#     y'' - τ y' + Δ y = 0
#
# the character of the solution is fixed by the roots of r² - τr + Δ = 0, i.e.
# by the discriminant τ² - 4Δ and the signs of τ and Δ:
#
#     Δ < 0                        saddle           (real roots, opposite signs)
#     Δ > 0, disc > 0, τ < 0       stable_node      (real roots, both negative)
#     Δ > 0, disc > 0, τ > 0       unstable_node    (real roots, both positive)
#     Δ > 0, disc < 0, τ < 0       stable_spiral    (complex, decaying)
#     Δ > 0, disc < 0, τ > 0       unstable_spiral  (complex, growing)
#     τ = 0, Δ > 0                 center           (purely imaginary)
#
# "train on one region, test on all six" is what produces the transfer matrices,
# so region-filtered sampling is the foundation the sweep and transfer
# experiments are built on.

const TRACE_DET_REGIONS = [:saddle, :stable_node, :unstable_node,
                           :stable_spiral, :unstable_spiral, :center]

"""
    region(tau, delta) → Symbol

Classify a point of the trace-determinant plane. Returns one of
`TRACE_DET_REGIONS`, or `:degenerate` (Δ = 0) / `:star` (disc = 0), which are
measure-zero boundaries the experiments exclude.
"""
function region(tau, delta)
  delta < 0 && return :saddle
  delta == 0 && return :degenerate
  disc = tau^2 - 4 * delta
  tau == 0 && return :center
  disc > 0 && return tau < 0 ? :stable_node : :unstable_node
  disc < 0 && return tau < 0 ? :stable_spiral : :unstable_spiral
  return :star
end

"""
    alpha_from_tau_delta(tau, delta) → Matrix{Float32}

The ODE coefficient matrix for `y'' - τy' + Δy = 0`.

Shape is `(ode_order+1) × (poly_degree+1)` = 3×1, matching this module's
convention that `α_matrix[k+1, j+1]` is the coefficient of `xʲ·u⁽ᵏ⁾`. These are
constant-coefficient ODEs, so there is a single column:

    α[1,1] = Δ    (u)
    α[2,1] = -τ   (u')
    α[3,1] = 1    (u'')

The 3×1 orientation matters: a 1×3 would mean ode_order 0 with polynomial
coefficients, i.e. `Δu - τx·u + x²·u = 0` — an algebraic equation with no
derivatives, which `solve_ode_series_closed_form` would happily consume and
return nonsense for. `vec()` is identical either way, which is why the loss
functions cannot detect the difference.

Inverse of `loss_functions.tau_delta_from_alpha`, so an ODE round-trips between
the two representations' input encodings.
"""
alpha_from_tau_delta(tau, delta) = reshape(Float32[delta, -tau, 1.0f0], 3, 1)

"""
    sample_region(reg, n; tau_lim, delta_lim, rng) → (taus, deltas)

Draw `n` points from a single trace-determinant region by rejection sampling
inside the box `[-tau_lim, tau_lim] × [-delta_lim, delta_lim]`.

`:center` is the τ = 0 axis — a measure-zero set that rejection sampling would
never hit — so it is sampled directly with τ = 0 and Δ > 0.
"""
function sample_region(reg::Symbol, n::Int;
                       tau_lim::Float32=2.0f0, delta_lim::Float32=2.0f0,
                       rng=Random.GLOBAL_RNG)
  reg in TRACE_DET_REGIONS || error(
    "unknown region :$reg; expected one of $(TRACE_DET_REGIONS)")
  taus = Float32[]
  deltas = Float32[]
  if reg === :center
    for _ in 1:n
      push!(taus, 0.0f0)
      push!(deltas, rand(rng, Float32) * delta_lim)
    end
  else
    while length(taus) < n
      t = rand(rng, Float32) * (2 * tau_lim) - tau_lim
      d = rand(rng, Float32) * (2 * delta_lim) - delta_lim
      if region(t, d) === reg
        push!(taus, t)
        push!(deltas, d)
      end
    end
  end
  return (taus, deltas)
end

"""
    sample_shell_region(reg, R, n; rng) → (taus, deltas)

Sample ODEs from a specific trace-determinant region within a shell:
`R-1 < max(|τ|, |Δ|) ≤ R`. Used by extrapolation and generalization-radius
experiments.
"""
function sample_shell_region(reg::Symbol, R::Real, n::Int; rng=Random.GLOBAL_RNG)
  reg in TRACE_DET_REGIONS || error(
    "unknown region :$reg; expected one of $(TRACE_DET_REGIONS)")
  taus = Float32[]
  deltas = Float32[]
  if reg === :center
    while length(taus) < n
      push!(taus, 0.0f0)
      push!(deltas, Float32(R - 1 + rand(rng, Float32)))
    end
  else
    while length(taus) < n
      t = rand(rng, Float32) * (2R) - R
      d = rand(rng, Float32) * (2R) - R
      m = max(abs(t), abs(d))
      if (Float32(R - 1) < m <= Float32(R)) && region(t, d) === reg
        push!(taus, t)
        push!(deltas, d)
      end
    end
  end
  return (taus, deltas)
end

"""
    generate_shell_dataset(reg, R, n, num_terms; a0, a1, rng) → Dict

Generate a dataset for shell `R` of a trace-determinant region.
Each entry maps the α-matrix `[Δ; -τ; 1]` to its derivative-basis series.
"""
function generate_shell_dataset(reg::Symbol, R::Real, n::Int, num_terms::Int;
                                a0::Float32=1.0f0, a1::Float32=0.0f0,
                                rng=Random.GLOBAL_RNG)
  taus, deltas = sample_shell_region(reg, R, n; rng=rng)
  dataset = Dict{Any,Any}()
  for (t, d) in zip(taus, deltas)
    α = alpha_from_tau_delta(t, d)
    series = solve_constant_coeff_series(α, [a0, a1], num_terms + 1)
    dataset[α] = Float32.(series[1:(num_terms + 1)])
  end
  return dataset
end

"""
    generate_grid_dataset(taus, deltas, num_terms; a0, a1) → Dict

Build a dataset from explicit (τ,Δ) pairs for grid evaluation.
Each entry maps the α-matrix to its derivative-basis series.
"""
function generate_grid_dataset(taus::Vector{<:Real}, deltas::Vector{<:Real},
                                num_terms::Int; a0::Float32=1.0f0, a1::Float32=0.0f0)
  length(taus) == length(deltas) || error("taus and deltas must have the same length")
  dataset = Dict{Any,Any}()
  for (t, d) in zip(taus, deltas)
    α = alpha_from_tau_delta(Float32(t), Float32(d))
    series = solve_constant_coeff_series(α, [a0, a1], num_terms + 1)
    dataset[α] = Float32.(series[1:(num_terms + 1)])
  end
  return dataset
end

"""
    solve_constant_coeff_series(α_matrix, initial_conditions, num_terms) → Vector{Float64}

Series coefficients for a CONSTANT-coefficient linear ODE, in the DERIVATIVE
basis (aₙ = u⁽ⁿ⁾(0)).

For `Σₖ αₖ u⁽ᵏ⁾ = 0` of order m, differentiating the equation n times gives the
recurrence directly in the derivative basis:

    a₍ₙ₊ₘ₎ = -(1/αₘ) · Σ_{k=0}^{m-1} αₖ · a₍ₙ₊ₖ₎

For 2nd order `u'' - τu' + Δu = 0` this is the familiar `a₍ₙ₊₂₎ = τa₍ₙ₊₁₎ - Δaₙ`.

WHY THIS EXISTS SEPARATELY FROM `solve_ode_series_closed_form`:
that function is intended for the general variable-coefficient case. Its
factorial handling was inconsistent — the numerator multiplied by a falling
factorial but the denominator did not divide by (n+m)!, producing coefficients
with magnitudes too large for stable PINN training. This function skips
factorials entirely, computing pure derivative-basis coefficients aₙ = u⁽ⁿ⁾(0),
which `monomial_from_derivative` converts to ψₙ = aₙ/n! at load time.

Requires a single column (no polynomial coefficients) and a nonzero leading
coefficient.
"""
function solve_constant_coeff_series(α_matrix, initial_conditions, num_terms::Int)
  rows, cols = size(α_matrix)
  cols == 1 || error(
    "solve_constant_coeff_series expects constant coefficients (1 column), got $cols. " *
    "Use solve_ode_series_closed_form for variable-coefficient ODEs.")
  m = rows - 1
  αm = Float64(α_matrix[m + 1, 1])
  abs(αm) > 1e-12 || error("leading coefficient α_$m is zero; ODE is not order $m")

  a = Float64.(collect(initial_conditions))
  length(a) >= m || error("need $m initial conditions for an order-$m ODE, got $(length(a))")

  while length(a) < num_terms
    n = length(a) - m          # index of the lowest term in this step
    s = 0.0
    for k in 0:(m - 1)
      s += Float64(α_matrix[k + 1, 1]) * a[n + k + 1]
    end
    push!(a, -s / αm)
  end
  return a[1:num_terms]
end

"""
    generate_region_dataset(regions, n_per_region, num_terms; kwargs...) → Dict

Build a `Dict{Matrix{Float32}, Vector{Float32}}` of
`alpha_matrix => series_coefficients` for the given regions — the same shape
`PINNSettings.ode_matrices` expects, so it drops straight into `train_pinn`.

Coefficients come from `solve_constant_coeff_series` (see its docstring for why
not `solve_ode_series_closed_form`). They are DERIVATIVE-basis aₙ; the loss
functions convert to the monomial basis ψₙ = aₙ/n! at buffer-build time.

`regions` may be a single Symbol or a collection.
"""
function generate_region_dataset(regions, n_per_region::Int, num_terms::Int;
                                 a0::Float32=1.0f0, a1::Float32=0.0f0,
                                 tau_lim::Float32=2.0f0, delta_lim::Float32=2.0f0,
                                 rng=Random.GLOBAL_RNG)
  regs = regions isa Symbol ? [regions] : collect(regions)
  dataset = Dict{Any,Any}()
  for reg in regs
    taus, deltas = sample_region(reg, n_per_region;
                                 tau_lim=tau_lim, delta_lim=delta_lim, rng=rng)
    for (t, d) in zip(taus, deltas)
      α = alpha_from_tau_delta(t, d)
      # num_terms + 1 total coefficients, seeded by the two initial conditions.
      series = solve_constant_coeff_series(α, [a0, a1], num_terms + 1)
      dataset[α] = Float32.(series[1:(num_terms + 1)])
    end
  end
  return dataset
end

# Factorial product - keep the same
function factorial_product_numeric(n_val, k, i)
  if k == 0
    return 1.0
  end
  product = 1.0
  for j in 1:k
    product *= (n_val + j - i)
  end
  return product
end

# General variable-coefficient ODE solver — returns derivative-basis coefficients aₙ = u⁽ⁿ⁾(0).
# No factorial terms here; monomial_from_derivative handles the ÷n! conversion downstream.
function solve_ode_series_closed_form(α_matrix, initial_conditions, num_terms)
  rows, cols = size(α_matrix)
  m = rows - 1  # ODE order

  # Initialize series with initial conditions
  series_coeffs = Float64.(initial_conditions)

  DEBUG && @info "DEBUG: Starting with initial conditions: $series_coeffs"
  DEBUG && @info "DEBUG: ODE order: $m"

  # Compute coefficients using closed form
  for n in 0:(num_terms-length(initial_conditions)-1)
    # Check if c_{m,0} is zero (would make equation singular)
    c_m_0 = α_matrix[m+1, 1]  # c_{m,0} is at position [m+1, 1]
    if c_m_0 == 0
      DEBUG && @warn "DEBUG: c_{m,0} = 0, cannot solve for this ODE"
      break
    end

    # Compute the sum term
    sum_term = 0.0

    for k in 0:m
      for j in 0:(cols-1)
        c_kj = α_matrix[k+1, j+1]  # c_{k,j} at position [k+1, j+1]

        if c_kj != 0
          coeff_index = n - j + k

          # Check if we have this coefficient available
          if coeff_index >= 0 && coeff_index < length(series_coeffs)
            # factorial_term = factorial_product_numeric(n - j, k, 0)
            # term_value = c_kj * factorial_term * series_coeffs[coeff_index+1] # factorial ON — larger magnitudes in loss
            term_value = c_kj * series_coeffs[coeff_index+1]
            sum_term += term_value
          end
        end
      end
    end

    # Apply the closed form formula: a_{n+m} = -(1/c_{m,0}) * sum
    # factorial_nm = factorial(big(n + m))  # unused — denominator does not divide by (n+m)!
    # denominator = c_m_0 * factorial_nm    # would give monomial basis, but monomial_from_derivative handles ÷n! downstream
    denominator = c_m_0

    new_coeff = -sum_term / denominator
    push!(series_coeffs, new_coeff)
  end

  return series_coeffs
end

# just generates the json file that you see in ./data
function generate_random_ode_dataset(s::Settings, batch_index::Int; exclude_matrix_keys::Set{String}=Set{String}(), exclude_canonical_keys::Set{String}=Set{String}(), coeff_bound::Int=1000)
  ode_order = s.ode_order
  poly_degree = s.poly_degree

  p_bar = Progress(s.dataset_size, desc="Generating ODEs...")

  # Generate dataset_size examples
  for example_k in 1:s.dataset_size
    # α_matrix = generate_random_alpha_matrix(s.ode_order, s.poly_degree) # generate ODE matrix
    α_matrix = generate_random_alpha_matrix_with_constraint(s.ode_order, s.poly_degree; coeff_bound=coeff_bound) # generate ODE matrix
    # Held-out guarantee: re-draw if this matrix is reserved (e.g. for the benchmark set).
    # Canonical-key exclusion also rejects scalar multiples of reserved ODEs — with
    # canonicalized network inputs they would be the identical training example.
    is_reserved(m) = string(m) in exclude_matrix_keys || canonical_matrix_key(m) in exclude_canonical_keys
    retries = 0
    while is_reserved(α_matrix)
      retries += 1
      retries > 1000 && error("Could not draw a matrix outside exclude_matrix_keys after 1000 retries — exclusion set may cover the whole matrix space")
      α_matrix = generate_random_alpha_matrix_with_constraint(s.ode_order, s.poly_degree; coeff_bound=coeff_bound)
    end

    # generate exactly ode_order initial conditions
    initial_conditions = Float64[]
    for i in 0:(ode_order-1)
      if i == 0
        # Fixed IC: y(0) = 1 — the map α → solution is only well-defined when the
        # IC is not a per-example random value the network can't see in its input.
        push!(initial_conditions, 1.0)  # y(0) = a_0
      elseif i == 1
        # push!(initial_conditions, 2.0)  # y'(0) = a_1, we will set thie init condition to be 1
        push!(initial_conditions, rand(1:5))  # y'(0) = a_1
      end
    end
    try
      # output taylor series and its coefficients
      # this actually computes the taylor series
      series_coeffs = solve_ode_series_closed_form(α_matrix, initial_conditions, s.num_of_terms)
      DEBUG && @info "DEBUG: Truncated series coefficients: $series_coeffs"

      # read existing data
      existing_data = if isfile(s.data_dir)
        JSON.parsefile(s.data_dir)
      else
        Dict()
      end

      # Determine which training run this is based on existing data
      dataset_key = lpad(batch_index, 2, '0')

      # Initialize dataset key if it doesn't exist
      if !haskey(existing_data, dataset_key)
        existing_data[dataset_key] = Dict()
      end

      # use alpha matrix as key, series coefficients as value within the dataset batch
      existing_data[dataset_key][string(α_matrix)] = series_coeffs # this is the source of our problems

      isdir("data") || mkpath("data") # ensure a data folder exists
      json_string = JSON.json(existing_data)
      write(s.data_dir, json_string)
    catch e
      @warn "Failed to solve this ODE" exception=e
      continue
    end

    ProgressMeter.next!(p_bar)
  end
end

# Generate a specific benchmark from one alpha matrix
function generate_specific_ode_dataset(s::Settings, batch_index::Int, α_matrix::Matrix{Int64})
  ode_order = s.ode_order
  poly_degree = s.poly_degree
  DEBUG && @info "DEBUG: Generating specific ODE benchmark" α_matrix
  # generate exactly ode_order initial conditions
  initial_conditions = Float64[]
  for i in 0:(ode_order-1)
    if i == 0
      # Fixed IC: y(0) = 1 — must match the training-data convention.
      push!(initial_conditions, 1.0)  # y(0) = a_0
    elseif i == 1
      # push!(initial_conditions, 2.0)  # y(0) = a_0, we will set thie init condition to be 1
      # push!(initial_conditions, rand(1:5))  # y'(0) = a_1
    end
  end
  try
    # output taylor series and its coefficients
    series_coeffs = solve_ode_series_closed_form(α_matrix, initial_conditions, s.num_of_terms)
    # read existing data
    existing_data = if isfile(s.data_dir)
      JSON.parsefile(s.data_dir)
    else
      Dict()
    end
    # Determine which training run this is based on existing data
    dataset_key = lpad(batch_index, 2, '0')
    # Initialize dataset key if it does not exist
    if !haskey(existing_data, dataset_key)
      existing_data[dataset_key] = Dict()
    end
    # use alpha matrix as key, series coefficients as value within the dataset batch
    #  α_matrix_key = join(["[" * join(row, ", ") * "]" for row in eachrow(α_matrix)], "; ")

    existing_data[dataset_key][string(α_matrix)] = series_coeffs # this is the source of our problems 
    isdir("data") || mkpath("data") # ensure a data folder exists
    json_string = JSON.json(existing_data)
    write(s.data_dir, json_string)
  catch e
    @warn "Failed to solve this ODE" exception=e
    return nothing
  end
end

# This function will be used for our experiment of taking scalar multiples of the coefficients of one ODE
function generate_ode_dataset_from_array_of_alpha_matrices(s::Settings, batch_index::Int, α_matrices::Array{Matrix{Int64}})
  ode_order = s.ode_order
  poly_degree = s.poly_degree
  # α_matrix = generate_random_alpha_matrix(s.ode_order, s.poly_degree) # generate ODE matrix
  # generate exactly ode_order initial conditions
  initial_conditions = Float64[]
  for i in 0:(ode_order-1)
    if i == 0
      # Fixed IC: y(0) = 1 — must match the training-data convention.
      push!(initial_conditions, 1.0)  # y(0) = a_0
      DEBUG && @info "DEBUG: y(0) = $(initial_conditions[end])"
    elseif i == 1
      push!(initial_conditions, rand(1:11))  # y'(0) = a_1
      DEBUG && @info "DEBUG: y'(0) = $(initial_conditions[end])"
    end
  end
  try
    for matrix in α_matrices
      series_coeffs = solve_ode_series_closed_form(matrix, initial_conditions, s.num_of_terms)
      DEBUG && @info "DEBUG: Truncated series coefficients: $series_coeffs"
      # read existing data
      existing_data = if isfile(s.data_dir)
        JSON.parsefile(s.data_dir)
      else
        Dict()
      end

      # Determine which training run this is based on existing data
      dataset_key = lpad(batch_index, 2, '0')
      # Initialize dataset key if it does not exist
      if !haskey(existing_data, dataset_key)
        existing_data[dataset_key] = Dict()
      end
      # use alpha matrix as key, series coefficients as value within the dataset batch
      #  α_matrix_key = join(["[" * join(row, ", ") * "]" for row in eachrow(α_matrix)], "; ")

      existing_data[dataset_key][string(matrix)] = series_coeffs # this is the source of our problems 
      isdir("data") || mkpath("data") # ensure a data folder exists
      json_string = JSON.json(existing_data)
      write(s.data_dir, json_string)
    end
  catch e
    @warn "Failed to solve this ODE" exception=e
    return nothing
  end
end

export Settings, generate_random_ode_dataset, generate_specific_ode_dataset, solve_ode_series_closed_form, generate_ode_dataset_from_array_of_alpha_matrices, canonical_matrix_key
export TRACE_DET_REGIONS, region, alpha_from_tau_delta, sample_region, sample_shell_region, generate_region_dataset, generate_shell_dataset, generate_grid_dataset, solve_constant_coeff_series
end
