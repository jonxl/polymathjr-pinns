#=
This script solves a third-order ordinary differential equation (ODE) using a
Physics-Informed Neural Network (PINN) inspired method.
Created based on the example available here: https://docs.sciml.ai/NeuralPDE/stable/examples/3rd/

Instead of the neural network approximating the solution u(x) directly, it learns
the optimal coefficients of a truncated power series that solves the ODE.

The process involves:
1. Defining the ODE and its boundary conditions.
2. Setting up a neural network that outputs a vector of coefficients.
3. Creating a loss function that measures how poorly the power series (built from the NN's coefficients)
   satisfies the ODE and boundary conditions.
4. Using an optimization algorithm (Adam) to train the network's parameters to minimize this loss.
5. Plotting the results to see how well our solution approximates the true, analytic solution.
=#

# ---------------------------------------------------------------------------
# Step 1: Import necessary libraries
# ---------------------------------------------------------------------------
# You may need to install these packages first. In the Julia REPL, press `]` to enter Pkg mode, then run:
# add Lux, ModelingToolkit, NeuralPDE, Optimization, OptimizationOptimJL, OptimizationOptimisers, Zygote, ComponentArrays, Plots, ProgressMeter

module PINN

struct PINNSettings
  Layers::Int
  Dimension::Int
  Output::Int
end

using Lux, ModelingToolkit
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote
using ComponentArrays
import IntervalSets: Interval
using Plots, ProgressMeter
import Random
using TaylorSeries

include("../utils/ProgressBar.jl")
using .ProgressBar

# Ensure a "data" directory exists for saving plots.
isdir("data") || mkpath("data")

# Define the floating point type to use throughout the script (e.g., Float32).
# Using Float32 is standard for neural networks as it's computationally faster.
F = Float32
# F = Float64

# ---------------------------------------------------------------------------
# Step 2: Define the Mathematical Problem (The ODE)
# ---------------------------------------------------------------------------

# Using ModelingToolkit, we define the independent variable `x` and the dependent variable `u(x)`.
@parameters x
@variables u(..)

# Do we compute x_right in loss_func??
x_left = F(0.0)  # Left boundary of the domain
x_right = F(1.0) # Right boundary of the domain

#=
# y(0) = 0
# y'(0) = 1
=#

# Dxxx = Differential(x)^3
Dx = Differential(x) # we are considering ay'+ ay = 0 with constant coefficients

# Define the ordinary differential equation.
# Dxxx(u(x)) = cos(pi*x)
# equation = Dxxx(u(x)) ~ cos(pi * x)

# Dx(u(x)) + u(x) = 0 (ay'+by = 0, a,b != 0)

#=
#
# Pass in the training data with the alpha_matrix
# So you have a vector of equations
#
=#

equation = Dx(u(x)) + u(x) ~ 0

# Define the boundary conditions for the ODE.
#= bcs = [u(0.0) ~ 0.0,        # u(x) at x=0 is 0
  u(1.0) ~ cos(pi),   # u(x) at x=1 is -1
  Dx(u(1.0)) ~ 1.0]      # The first derivative u'(x) at x=1 is 1
=#

# Define the domain over which the ODE is valid.
domains = [x ∈ Interval(x_left, x_right)]

# For verification, we define the true, known analytic solution to the ODE.
# This will be used to calculate the error of our approximation.
# analytic_sol_func(x) = (pi * x * (-x + (pi^2) * (2x - 3) + 1) - sin(pi * x)) / (pi^3) # We replace with our training examples
# TODO: Put the training data
# benchmark baby!!!!
# This could be a validator function

# ---------------------------------------------------------------------------
# Step 3: Setup the Power Series and Neural Network
# ---------------------------------------------------------------------------

# We will approximate the solution u(x) with a truncated power series of degree N.
# u(x) ≈ a₀ + a₁x/1! + a₂x²/2! + ... + aₙxⁿ/N!
N = 1 # The degree of the highest power term in the series.

# Pre-calculate factorials (0!, 1!, ..., N!) for use in the series.
fact = factorial.(0:N)

num_supervised = 1 # The number of coefficients we will supervise during training.
supervised_weight = F(1.0)  # Weight for the supervised loss term in the total loss function.

#= This is where I have to replace the approximation to the ODE with the
coefficients generated from the plugboardmethod =#
# The true coefficients a_n are the n-th derivatives of the analytic solution at x=0.
# We approximate them using TaylorSeries.jl.
# t = Taylor1(F, N)
# taylor_expansion = analytic_sol_func(t)
# a_true = taylor_expansion.coeffs .* fact
# training_data = a_true[1:num_supervised] # replace this with the plugboard coefficients

# Create a set of points inside the domain to enforce the ODE. These are called "collocation points".
num_points = 1000
xs = range(x_left, x_right, length=num_points)

# Define a weight for the boundary condition part of the loss. This is a hyperparameter
# that helps balance the importance of satisfying the ODE vs. the boundary conditions.
bc_weight = F(100.0)

# Define the neural network architecture using Lux.
# It takes one dummy input and outputs N+1 values, which will be our coefficients a₀ to aₙ.
# coeff_net = Lux.Chain(
#   Lux.Dense(1, 64, σ), # Hidden layer with 100 neurons and sigmoid activation.
#   Lux.Dense(64, N+1)             # Output layer with N+1 neurons (one for each coefficient).
# )

#=
nrows, ncols = size(A)
rows = size(A)[1]
cols = size(A)[2]
=#

coeff_net = Lux.Chain(
  Lux.Dense(size(ode_matrix), 64, σ), # Hidden layer with 64 neurons and tanh activation.
  # Lux.Dense(64, 64, σ), # Hidden layer with 64 neurons and tanh activation.
  # Lux.Dense(64, 64, σ), # Hidden layer with 64 neurons and tanh activation.
  Lux.Dense(64, N + 1)      # Output layer with N+1 neurons (one for each coefficient).
)

# Initialize the network's parameters (p) and state (st).
rng = Random.default_rng()
Random.seed!(rng, 1234) # Seeding for reproducibility.
p_init, st = Lux.setup(rng, coeff_net)

# Wrap the initial parameters in a `ComponentArray`. This makes the nested `NamedTuple` of
# parameters behave like a standard vector, making it compatible with the optimizer.
p_init_ca = ComponentArray(p_init) # wtf? 


# ---------------------------------------------------------------------------
# Step 4: Define the Loss Function
# ---------------------------------------------------------------------------

# TODO: ADJUST THIS FOR ODE of first ORDER
# This loss function is more or less the same for training and benchmarking

# The loss function measures how "wrong" our current approximation is. The optimizer's
# job is to find the network parameters `p_net` that minimize this function.

# generic loss function
function loss_fn(p_net, ode_matrix, data) # for loop in here
  # First, run the network to get the current vector of power series coefficients.
  # The network takes a dummy input [0.0] and outputs our N+1 coefficients.
  a_vec = first(coeff_net([ode_matrix], p_net, st))[:, 1] # how do we adjust for a matrix

  # Define the approximate solution and its derivatives using the coefficients from the network.
  # This is the power series representation: u(x) = Σ aᵢ * x^(i-1) / (i-1)!
  u_approx(x) = sum(a_vec[i] * x^(i - 1) / fact[i] for i in 1:N+1)
  Du_approx(x) = sum(a_vec[i] * x^(i - 2) / fact[i-1] for i in 2:N+1) # First derivative
  # D3u_approx(x) = sum(a_vec[i] * x^(i - 4) / fact[i-3] for i in 4:N+1) # Third derivative

  # Calculate the loss from the ODE itself (the PDE loss, though it's an ODE).
  # This is the mean squared error between the two sides of our equation over all collocation points.
  loss_pde = sum(abs2, Du_approx(xi) + u_approx(xi) - 0 for xi in xs) / num_points # adjust this

  # Calculate the loss from the boundary conditions.
  # This is the sum of squared errors for each boundary condition.
  loss_bc = abs2(u_approx(x_left) - F(0.0))
  # abs2(u_approx(x_right) - F(cos(pi))) +
  # abs2(Du_approx(x_right) - F(1.0)) # we might need this

  loss_supervised = sum(abs2, a_vec[1:num_supervised] - data) / num_supervised # training_data from Plugboard
  # loss_supervised = 0.0 # This is a placeholder for any supervised loss, if needed.
  # The total loss is a weighted sum of the two components.
  return loss_pde + bc_weight * loss_bc + supervised_weight * loss_supervised
end

# GLOBAL LOSS
function global_loss()
  # TODO:
  # for loop through plugboard coefficients
  # the sum of the local loss 

end


# ---------------------------------------------------------------------------
# Step 5: Train the Neural Network
# ---------------------------------------------------------------------------

# ---------------- stage 1 : ADAM ----------------

#=
# Adam training process:
# - Cheaper
# - More iterations
# 
=#

maxiters = 500 # The total number of training iterations.
p_one::ProgressBarSettings = ProgressBar.ProgressBarSettings(maxiters, "Adam Training...") # first progress bar setting
callback_one = ProgressBar.Bar(p_one)
# Define the optimization problem. We specify the loss function, the initial parameters,
# and the automatic differentiation backend (Zygote), which calculates the gradients.
adtype = Optimization.AutoZygote()
optfun = OptimizationFunction(loss_fn, adtype)
prob = OptimizationProblem(optfun, p_init_ca) # component array NEEDS to be used

# Now, we solve the problem using the Adam optimizer.
res = solve(prob, # solve is a general function for an optimizer
  OptimizationOptimisers.Adam(F(1e-3)); # Adam optimizer with a learning rate of 0.001
  callback=callback_one,
  maxiters=maxiters)

# ---------------- stage 2 : LBFGS ----------------
#=
# LBFGS training process:
# - More expensive
# - Newton's method 2nd Hessian
# - Less training runs
=#



maxiters_lbfgs = 100
p_two::ProgressBarSettings = ProgressBar.ProgressBarSettings(maxiters_lbfgs, "LBFGS fine-tune... ")
callback_two = ProgressBar.Bar(p_two)

prob2 = remake(prob; u0=res.u)
res = solve(prob2,
  OptimizationOptimJL.LBFGS();
  callback=callback_two,
  maxiters=maxiters_lbfgs)

# Extract the final, trained parameters from the result.
p_trained = res.u
# Run the network one last time with the trained parameters to get the final coefficients.
a_learned = first(coeff_net([F(0.0)], p_trained, st))[:, 1]
# TODO: for loop the PINN function against the vectors in TRAINING set
# PINN would have to be a function

# ---------------------------------------------------------------------------
# Step 6: Analyze and Plot the Results
# ---------------------------------------------------------------------------

println("\nTraining complete.")
println("Learned coefficients:")
display(a_learned)

# --- Plot 1: Compare learned solution vs. analytic solution ---

# Create a function for our predicted solution using the learned coefficients.
u_predict_func(x) = sum(a_learned[i] * x^(i - 1) / fact[i] for i in 1:N+1)

# Generate a dense set of points for smooth plotting.
x_plot = x_left:F(0.01):x_right
u_real = analytic_sol_func.(x_plot) # replaced with the plugboard coefficients
# TODO: Taylor Series with plugboard coefficients dataset["01"][alpha_matrix]
u_predict = u_predict_func.(x_plot)

# --- Plot 1: Plot the actual solution ---
# TODO: actual solution plotted from Taylor Series
plot_compare = plot(x_plot, u_real, label="Analytic Solution", linestyle=:dash, linewidth=3)
plot!(plot_compare, x_plot, u_predict, label="PINN Power Series", linewidth=2)
title!(plot_compare, "ODE Solution Comparison")
xlabel!(plot_compare, "x")
ylabel!(plot_compare, "u(x)")
savefig(plot_compare, "data/solution_comparison.png")

# --- Plot 2: Plot the absolute error of the solution ---
error = max.(abs.(u_real .- u_predict), F(1e-20))
plot_error = plot(x_plot, error,
  title="Absolute Error of Power Series Solution",
  label="|Analytic - Predicted|",
  yscale=:log10, # Use a logarithmic scale for the y-axis
  xlabel="x",
  ylabel="Error",
  linewidth=2)
savefig(plot_error, "data/error.png")

# --- Plot 3: Plot the error of the learned coefficients ---

# TODO: pass in the coefficients not the taylor series
# Calculate the absolute error between the true and learned coefficients.
# Clamp the error to a minimum value to avoid log(0) issues.
coeff_error = max.(abs.(a_true - a_learned), F(1e-20)) # COMPUTES THE ERROR!!!!

plot_coeffs = plot(0:N, coeff_error,
  yscale=:log10, # Use a logarithmic scale for the y-axis
  title="Error in Learned Coefficients (log scale)",
  xlabel="Coefficient Index",
  ylabel="Absolute Error",
  label="Coefficient Error",
  legend=:topright)
savefig(plot_coeffs, "data/coefficient_error.png")


println("\nPlots saved to 'data' directory.")
println("- solution_comparison.png")
println("- error.png")
println("- coefficient_error.png")


export global_loss, PINNSettings

end
