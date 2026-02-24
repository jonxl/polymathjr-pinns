module helper_funcs

using CSV
using DataFrames
using Dates
using Plots
using Random
using JSON

function convert_plugboard_keys(inner_dict)
  converted_dict = Dict{Matrix{Int},Any}()
  for (alpha_matrix_key, series_coeffs) in inner_dict
    # println("The current  ODE I am calculating the loss for right now: ", alpha_matrix_key)
    # println("The local loss is locally lossing...")
    alpha_matrix = eval(Meta.parse(alpha_matrix_key)) # convert from string to matrix 
    converted_dict[alpha_matrix] = series_coeffs
  end

  return converted_dict
end

function show_benchmark_loss(
  array_of_benchmark_loss,
  filename,
  title,
  color,
  bar_width
)

  # Create bar graph from array
  bar(
    1:length(array_of_benchmark_loss),
    array_of_benchmark_loss,
    xlabel = "Training Run",
    ylabel = "Function Error",
    title = title,
    legend = false,
    grid = true,
    bar_width = bar_width,
    color = color
  )

  # Save the plot
  savefig(filename)
end

# this will be a reference for our new analytic_sol_fun
# analytic_sol_func(x) = (pi * x * (-x + (pi^2) * (2x - 3) + 1) - sin(pi * x)) / (pi^3) # We replace with our training examples

"""
This function will give us all the types of solutions for the homogenous ODE y'' + ay' + by = 0
depending on the discriminats value
"""

#=

function exponential_solution(x, c1, c2, α, β, D)
  if D > 0
    return c1 * exp(α * x) + c2 * exp(β * x)
  elseif D = 0
    return (c1 + c2 * x) * exp((-α * x)/2)
  elseif D < 0
    return exp(α * x) * (c1 * cos(β * x) + c2 * sin(β * x))
  end
end

=#

"""
Quadratic formula, this will be used for our analytic function we have. This will determine the roots of the function
"""
function quadratic_formula(a, b, c)
  discr = b^2 - 4*a*c
  # Handle complex roots if discriminant is negative
  sq = (discr > 0) ? sqrt(discr) : sqrt(discr + 0im)
  roots = [(-b - sq) / (2a), (-b + sq) / (2a)]
  return sort(roots, by=real, rev=true)  # Sort by real part, descending
end


# Generate random values for a and b that satisfy a² - 4b > 0
# Strategy: choose a first, then choose b such that b < a²/4
function generate_valid_matrix()
  a = rand(-10:10)  # Random integer for a
  max_b_float = (a^2) / 4 - 1  # Subtract 1 to ensure strict inequality with integers
  max_b = floor(Int, max_b_float)  # Convert to integer

  # Only generate b if max_b is positive
  if max_b > 0
    b = rand(0:max_b)
  else
    b = 0
  end

  return reshape([1, a, b], 3, 1)  # 3x1 column matrix
end

function create_matrix_array(n::Int)
  array_of_matrices = Matrix{Int64}[]

  for i in 1:n
    valid_matrix = generate_valid_matrix()

    # Verify the constraint (optional, for debugging)
    a = valid_matrix[2, 1]
    b = valid_matrix[3, 1]
    @assert a^2 - 4*b > 0 "Constraint violated: a² - 4b = $(a^2 - 4*b)"

    push!(array_of_matrices, valid_matrix)
  end

  return array_of_matrices
end


"""
Generate a unique run ID like "adam-02-23-26"
"""
function generate_run_id(optimizer::String)::String
  timestamp = Dates.format(now(), "mm-dd-yy")
  return "$(optimizer)-$(timestamp)"
end

"""
Append a results entry to results.json as an array.
Handles migration from old single-dict format to array format.
"""
function append_to_results_json(results_file, run_id, results_dict)
  # Read existing data or start fresh
  entries = []
  if isfile(results_file)
    existing = JSON.parsefile(results_file)
    if existing isa Vector
      entries = existing
    elseif existing isa Dict
      # Migrate old single-dict format into array
      push!(entries, existing)
    end
  end

  # Add the new entry with its run ID
  results_dict["id"] = run_id
  push!(entries, results_dict)

  # Ensure directory exists
  dir = dirname(results_file)
  if !isempty(dir) && !isdir(dir)
    mkpath(dir)
  end

  # Write back
  open(results_file, "w") do io
    JSON.print(io, entries, 2)
  end
end

export convert_plugboard_keys, exponential_solution, quadratic_formula, generate_valid_matrix, create_matrix_array, generate_run_id, append_to_results_json
end
