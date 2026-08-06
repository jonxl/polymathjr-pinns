#=
SINGLE-ODE memorization test.

Same PINN core as exp_family_sum.jl / generalize_pinn.jl (identical net, loss, and
Adam->LBFGS optimizer), but the training set is ONE exponential ODE instead of 640.
Then we test on that SAME ODE.

        y'' - tau*y' + Delta*y = 0,    y(0)=1, y'(0)=0
        net : (tau, Delta) -> [c_0, ..., c_N]      (u(x) = sum c_n x^n)

With a single ODE the (tau,Delta) input is constant, so the network is just an
over-parametrized way to produce one coefficient vector (the original PINN_specific.jl
setup). We expect it to MEMORIZE this ODE to near-zero error -- in contrast to the ~5%
the shared family net reaches even on a training ODE.

Loss (same knobs as the family scripts):
    loss = pde_weight * loss_pde + ic_weight * loss_ic + sup_weight * loss_sup
=#

using Lux
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote, ComponentArrays
using Plots, ProgressMeter
import Random
using LinearAlgebra

isdir("data") || mkpath("data")
F = Float32
Random.seed!(1234)

# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------
N  = 12
a0 = F(1.0); a1 = F(0.0)
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)

# the single ODE to memorize (saddle: roots +/-1, so u(x)=cosh(x))
τ0 = F(0.0); Δ0 = F(-1.0)

x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))

adam_iters  = 3000
lbfgs_iters = 2000

# ---------------------------------------------------------------------------
# Helpers (same as the family scripts)
# ---------------------------------------------------------------------------
fact_vec = F.(factorial.(0:N))
function true_cn(tau, delta)
    a = zeros(F, N + 1); a[1] = a0; a[2] = a1
    for n in 0:(N - 2)
        a[n + 3] = tau * a[n + 2] - delta * a[n + 1]
    end
    return a ./ fact_vec
end

# single-column dataset
X = reshape(F[τ0, Δ0], 2, 1)            # 2 × 1
Y = reshape(true_cn(τ0, Δ0), N + 1, 1)  # (N+1) × 1
nb = 1
println("training on ONE ODE: (τ,Δ)=($τ0,$Δ0)")

Pu  = F[ xs[m]^(i - 1)                                    for m in 1:n_colloc, i in 1:N+1 ]
Pu1 = F[ (i - 1) >= 1 ? (i - 1) * xs[m]^(i - 2) : 0       for m in 1:n_colloc, i in 1:N+1 ]
Pu2 = F[ (i - 1) >= 2 ? (i - 1)*(i - 2)*xs[m]^(i - 3) : 0 for m in 1:n_colloc, i in 1:N+1 ]

# ---------------------------------------------------------------------------
# Network: (tau, Delta) -> c_n   (same architecture as the family scripts)
# ---------------------------------------------------------------------------
net = Lux.Chain(Lux.Dense(2, 64, tanh), Lux.Dense(64, 64, tanh),
                Lux.Dense(64, 64, tanh), Lux.Dense(64, N + 1))
p_init, st = Lux.setup(Random.default_rng(), net)
p_init_ca  = ComponentArray(p_init)

function pinn_loss(p)
    C  = first(net(X, p, st))
    τ  = X[1:1, :]; Δ = X[2:2, :]
    resid    = (Pu2 * C) .- τ .* (Pu1 * C) .+ Δ .* (Pu * C)
    loss_pde = sum(abs2, resid) / (n_colloc * nb)
    loss_ic  = (sum(abs2, C[1, :] .- a0) + sum(abs2, C[2, :] .- a1)) / nb
    loss_sup = sup_weight == 0 ? zero(F) : sum(abs2, C .- Y) / ((N + 1) * nb)
    return pde_weight * loss_pde + ic_weight * loss_ic + sup_weight * loss_sup
end
loss_fn(p, _) = pinn_loss(p)

function rel_rmse(p)
    C = first(net(X, p, st))
    sqrt(sum(abs2, C .- Y) / (sum(abs2, Y) + F(1e-8)))
end

# the three loss terms returned SEPARATELY (raw / unweighted) for the breakdown plot
function loss_components(p)
    C  = first(net(X, p, st))
    τ  = X[1:1, :]; Δ = X[2:2, :]
    lp = sum(abs2, (Pu2 * C) .- τ .* (Pu1 * C) .+ Δ .* (Pu * C)) / (n_colloc * nb)
    li = (sum(abs2, C[1, :] .- a0) + sum(abs2, C[2, :] .- a1)) / nb
    ls = sum(abs2, C .- Y) / ((N + 1) * nb)
    return lp, li, ls
end

# ---------------------------------------------------------------------------
# Train (Adam -> LBFGS)
# ---------------------------------------------------------------------------
hist = Float64[]
pde_hist = Float64[]; ic_hist = Float64[]; sup_hist = Float64[]
function log_comp!(p)
    lp, li, ls = loss_components(p)
    push!(pde_hist, lp); push!(ic_hist, li); push!(sup_hist, ls)
end
p_bar = Progress(adam_iters, desc = "single-ODE Adam ")
i1 = 0
callback = function (state, l)
    global i1 += 1; push!(hist, rel_rmse(state.u)); log_comp!(state.u)
    ProgressMeter.next!(p_bar; showvalues = [(:iter, i1), (:loss, l), (:relerr, hist[end])])
    return false
end
prob = OptimizationProblem(OptimizationFunction(loss_fn, Optimization.AutoZygote()), p_init_ca)
res  = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); callback = callback, maxiters = adam_iters)

n_adam = length(hist)
p_bar2 = Progress(lbfgs_iters, desc = "single-ODE LBFGS ")
i2 = 0
cb2 = function (state, l)
    global i2 += 1; push!(hist, rel_rmse(state.u)); log_comp!(state.u)
    ProgressMeter.next!(p_bar2; showvalues = [(:iter, i2), (:loss, l), (:relerr, hist[end])])
    return false
end
res = solve(remake(prob; u0 = res.u), OptimizationOptimJL.LBFGS(); callback = cb2, maxiters = lbfgs_iters)
p_trained = res.u

# ---------------------------------------------------------------------------
# Evaluate on the SAME ODE
# ---------------------------------------------------------------------------
c_pred = vec(first(net(X, p_trained, st)))
c_true = vec(Y)
e_final = sqrt(sum(abs2, c_pred .- c_true) / (sum(abs2, c_true) + 1e-12))

println("\nweights: pde=$pde_weight ic=$ic_weight sup=$sup_weight")
println("trained & tested on (τ,Δ)=($τ0,$Δ0)")
println("coefficient rel-error (same ODE) = ", round(e_final; sigdigits = 4))
println("final loss components (raw):  pde=$(round(pde_hist[end];sigdigits=3))  ",
        "ic=$(round(ic_hist[end];sigdigits=3))  sup=$(round(sup_hist[end];sigdigits=3))")

xfine = collect(range(x_left, x_right, length = 200))
eval_series(c) = [sum(c[i] * x^(i - 1) for i in 1:N+1) for x in xfine]

# analytic solution (exact, not truncated): c1 e^{λ1 x} + c2 e^{λ2 x} with ICs (a0,a1)
function u_analytic(xpts)
    s  = sqrt(Float64(τ0)^2 - 4 * Float64(Δ0))
    λ1 = (Float64(τ0) + s) / 2; λ2 = (Float64(τ0) - s) / 2
    c1 = (Float64(a1) - λ2 * Float64(a0)) / (λ1 - λ2); c2 = Float64(a0) - c1
    return [c1 * exp(λ1 * x) + c2 * exp(λ2 * x) for x in xpts]
end

# --- error measured two ways ---
u_pred_grid = eval_series(c_pred); u_true_grid = u_analytic(xfine)
sol_relL2 = sqrt(sum(abs2, u_pred_grid .- u_true_grid) / sum(abs2, u_true_grid))   # solution-space rel-L2
sol_Linf  = maximum(abs.(u_pred_grid .- u_true_grid))                              # solution-space max-abs (repo-style)
println("\n--- error measured two ways ---")
println("coefficient space  (rel-L2 vs truncated Taylor cₙ) = ", round(e_final;   sigdigits = 4))
println("solution   space   (rel-L2 vs analytic u)          = ", round(sol_relL2; sigdigits = 4))
println("solution   space   (max-abs / L∞, repo-style)      = ", round(sol_Linf;  sigdigits = 4))

plot_loss = plot(1:length(hist), max.(hist, 1e-20), yscale = :log10, lw = 2, legend = false,
                 title = "single-ODE memorization: rel-err vs iteration",
                 xlabel = "Iteration (Adam then LBFGS)", ylabel = "rel-RMSE")
vline!(plot_loss, [n_adam + 0.5], ls = :dash, color = :gray)
savefig(plot_loss, "data/single_ode_loss.png")

plot_sol = plot(title = "single ODE (τ,Δ)=($τ0,$Δ0)  rel-err=$(round(e_final;sigdigits=3))",
                xlabel = "x", ylabel = "u(x)", legend = :topleft)
plot!(plot_sol, xfine, eval_series(c_pred), lw = 2, color = 1, label = "pred")
plot!(plot_sol, xfine, eval_series(c_true), lw = 1, ls = :dash, color = 1, label = "true")
savefig(plot_sol, "data/single_ode_solution.png")

# --- repo-style absolute solution error |u_pred - u_analytic| vs x ---
plot_solerr = plot(xfine, max.(abs.(u_pred_grid .- u_true_grid), 1e-20), yscale = :log10, lw = 2,
                   legend = false, title = "single-ODE: absolute solution error |u_pred − u_analytic|",
                   xlabel = "x", ylabel = "|error|")
savefig(plot_solerr, "data/single_ode_solerr.png")

# --- loss breakdown: the three terms (raw, unweighted) vs iteration ---
its = 1:length(pde_hist)
plot_comp = plot(its, max.(pde_hist, 1e-20), yscale = :log10, lw = 2, label = "pde residual",
                 title = "single-ODE memorization: loss breakdown (raw, unweighted)",
                 xlabel = "Iteration (Adam then LBFGS)", ylabel = "component value")
plot!(plot_comp, its, max.(ic_hist,  1e-20), lw = 2, label = "ic")
plot!(plot_comp, its, max.(sup_hist, 1e-20), lw = 2, label = "supervised")
vline!(plot_comp, [n_adam + 0.5], ls = :dash, color = :gray, label = "Adam | LBFGS")
savefig(plot_comp, "data/single_ode_components.png")

println("Plots: data/single_ode_loss.png , data/single_ode_solution.png , data/single_ode_solerr.png , data/single_ode_components.png")
