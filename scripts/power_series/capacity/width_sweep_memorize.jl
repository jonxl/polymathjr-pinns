#=
WIDTH SWEEP -- single-ODE memorization vs network width.

Same single-ODE memorization test as single_ode_memorize.jl (train on ONE exponential
ODE, test on that SAME ODE), repeated across a range of hidden widths -- both SMALLER
and BIGGER than the default 64. Architecture is fixed at 3 hidden layers; only the width
changes:  Dense(2 -> W -> W -> W -> N+1), tanh.

Question: does memorization of a single ODE depend on capacity? Since one ODE needs only
N+1 = 13 output numbers, we expect even small nets to memorize it near the floor -- i.e.
width should barely matter here (capacity is NOT the bottleneck for one target). Contrast
with the family case, where capacity is shared across many ODEs.

Same loss (pde + ic + sup, all weights 1) and Adam->LBFGS optimizer as the other scripts.
=#

using Lux
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote, ComponentArrays
using Plots, ProgressMeter
import Random
using Random: MersenneTwister
using LinearAlgebra

isdir("data") || mkpath("data")
F = Float32

# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------
N  = 12
a0 = F(1.0); a1 = F(0.0)
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)

τ0 = F(0.0); Δ0 = F(-1.0)        # the single ODE to memorize (saddle, u=cosh x)

x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))

adam_iters  = 3000
lbfgs_iters = 2000

widths = [4, 8, 16, 32, 64, 128, 256]

# ---------------------------------------------------------------------------
# Helpers / data (single column)
# ---------------------------------------------------------------------------
fact_vec = F.(factorial.(0:N))
function true_cn(tau, delta)
    a = zeros(F, N + 1); a[1] = a0; a[2] = a1
    for n in 0:(N - 2)
        a[n + 3] = tau * a[n + 2] - delta * a[n + 1]
    end
    return a ./ fact_vec
end

X = reshape(F[τ0, Δ0], 2, 1)
Y = reshape(true_cn(τ0, Δ0), N + 1, 1)
nb = 1

Pu  = F[ xs[m]^(i - 1)                                    for m in 1:n_colloc, i in 1:N+1 ]
Pu1 = F[ (i - 1) >= 1 ? (i - 1) * xs[m]^(i - 2) : 0       for m in 1:n_colloc, i in 1:N+1 ]
Pu2 = F[ (i - 1) >= 2 ? (i - 1)*(i - 2)*xs[m]^(i - 3) : 0 for m in 1:n_colloc, i in 1:N+1 ]

# loss / metric parametrized by (net, st) so we can swap widths
function pinn_loss(p, net, st)
    C  = first(net(X, p, st))
    τ  = X[1:1, :]; Δ = X[2:2, :]
    resid    = (Pu2 * C) .- τ .* (Pu1 * C) .+ Δ .* (Pu * C)
    loss_pde = sum(abs2, resid) / (n_colloc * nb)
    loss_ic  = (sum(abs2, C[1, :] .- a0) + sum(abs2, C[2, :] .- a1)) / nb
    loss_sup = sup_weight == 0 ? zero(F) : sum(abs2, C .- Y) / ((N + 1) * nb)
    return pde_weight * loss_pde + ic_weight * loss_ic + sup_weight * loss_sup
end
rel_rmse(p, net, st) = sqrt(sum(abs2, first(net(X, p, st)) .- Y) / (sum(abs2, Y) + F(1e-8)))

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
errs   = Float64[]
nparam = Int[]
for W in widths
    net = Lux.Chain(Lux.Dense(2, W, tanh), Lux.Dense(W, W, tanh),
                    Lux.Dense(W, W, tanh), Lux.Dense(W, N + 1))
    rng = MersenneTwister(1234)          # same seed each width for comparability
    p0, st = Lux.setup(rng, net)
    p0ca = ComponentArray(p0)
    push!(nparam, length(p0ca))

    lf(p, _) = pinn_loss(p, net, st)
    prob = OptimizationProblem(OptimizationFunction(lf, Optimization.AutoZygote()), p0ca)
    pb = Progress(adam_iters, desc = "W=$W Adam ")
    cb = (s, l) -> (next!(pb; showvalues = [(:loss, l), (:relerr, rel_rmse(s.u, net, st))]); false)
    r = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); callback = cb, maxiters = adam_iters)
    r = solve(remake(prob; u0 = r.u), OptimizationOptimJL.LBFGS(); maxiters = lbfgs_iters)

    e = rel_rmse(r.u, net, st)
    push!(errs, e)
    println("  width=$W  params=$(nparam[end])  rel-err=$(round(e; sigdigits = 4))")
end

# ---------------------------------------------------------------------------
# Report + plot
# ---------------------------------------------------------------------------
println("\nsingle-ODE memorization vs width  (τ,Δ)=($τ0,$Δ0)")
println(rpad("width", 10), rpad("params", 12), "rel-err")
for k in eachindex(widths)
    println(rpad(widths[k], 10), rpad(nparam[k], 12), round(errs[k]; sigdigits = 4))
end

p = plot(widths, max.(errs, 1e-20), xscale = :log10, yscale = :log10,
         marker = :circle, lw = 2, legend = false,
         title = "Single-ODE memorization vs width  (train=test, 1 ODE)",
         xlabel = "hidden width W", ylabel = "rel-RMSE (same ODE)")
savefig(p, "data/width_sweep_memorize.png")
println("\nPlot: data/width_sweep_memorize.png")
