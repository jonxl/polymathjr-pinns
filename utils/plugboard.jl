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
function generate_random_alpha_matrix_with_constraint(ode_order, poly_degree; coeff_bound::Int=1000, max_ratio::Int=10)
  rows = ode_order + 1
  cols = poly_degree + 1
  # First-order case (2x1 matrix [α₀; α₁] for α₀·y + α₁·y' = 0):
  # draw nonzero integers from a wide ±coeff_bound range so the matrix space is
  # large, but cap |α₀/α₁| ≤ max_ratio — the solution is y = A·e^{-(α₀/α₁)x} and
  # its series coefficients grow like (α₀/α₁)^n, which overflows Float32 training
  # if the ratio is unbounded.
  if rows == 2 && cols == 1
    rand_nonzero() = rand(Bool) ? rand(-coeff_bound:-1) : rand(1:coeff_bound)
    α0 = rand_nonzero()
    α1 = rand_nonzero()
    while abs(α0) > max_ratio * abs(α1)
      α0 = rand_nonzero()
      α1 = rand_nonzero()
    end
    return reshape([α0, α1], 2, 1)
  # Special handling for the constraint case (3x1 matrix)
  elseif rows == 3 && cols == 1
    # Generate column matrix [1, a, b]^T with constraint a² - 4b > 0
    constraint_satisfied = false
    a = 0
    b = 0
    while !constraint_satisfied
      a = rand(Bool) ? rand(-10:-1) : rand(1:10)
      max_b_float = (a^2) / 4
      max_b = floor(Int, max_b_float)
      # Choose b to satisfy the constraint (excluding zero)
      if max_b >= 1
        b = rand(1:max_b)
      elseif max_b >= -10 && max_b < 0
        b = rand(max_b:-1)
      else
        b = rand(-10:-1)
      end
      # Check if constraint is satisfied and b is not zero
      if a^2 - 4*b > 0 && b != 0
        constraint_satisfied = true
      end
    end
  return reshape([1, a, b], 3, 1) 
  else
    # Original behavior for other matrix sizes
    α_matrix = Matrix{Int}(undef, rows, cols)
    for i in 1:rows
      for j in 1:cols
        α_matrix[i, j] = rand(Bool) ? rand(-10:-1) : rand(1:10)
      end
    end
    return α_matrix
  end
end

# Canonical form of an ODE matrix: divide by the leading (last) coefficient,
# mirroring helper_funcs.canonicalize_alpha. Scalar multiples of the same ODE
# share one canonical key — used for held-out exclusion.
canonical_matrix_key(α_matrix) = string(Float32.(vec(α_matrix) ./ vec(α_matrix)[end]))

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

# New closed-form implementation
# this just 
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
            factorial_term = factorial_product_numeric(n - j, k, 0)
            term_value = c_kj * factorial_term * series_coeffs[coeff_index+1] # is this what they mean by SHUT OFF THE FACTORIAL !!!!
            # term_value = c_kj * series_coeffs[coeff_index+1] # this is when we want factorial_tern off.
            sum_term += term_value
          end
        end
      end
    end

    # Apply the closed form formula: a_{n+m} = -(1/(c_{m,0} * (n+m)!)) * sum
    factorial_nm = factorial(big(n + m))  # Use big integer for large factorials
    # denominator = c_m_0 * factorial_nm # no this is where it is
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
    series_coeffs = solve_ode_series_closed_form(α_matrix, initial_conditions, s.num_of_terms) # haha this was the issue
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
      series_coeffs = solve_ode_series_closed_form(matrix, initial_conditions, s.num_of_terms) # haha this was the issue 
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
end
