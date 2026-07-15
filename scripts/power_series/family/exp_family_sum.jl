#=
EXPONENTIAL FAMILY: recover two distinct ODEs and their sum.

We train one PINN over the "exponential" members of the 2nd-order linear family
(those with REAL eigenvalues -- saddle / stable node / unstable node; the spiral &
center regions are oscillatory, NOT exponential, so they are excluded).

        y'' - tau*y' + Delta*y = 0,    y(0)=1, y'(0)=0
        net : (tau, Delta) -> [c_0, ..., c_N]      (u(x) = sum c_n x^n)

(Same proven setup as generalize_pinn.jl -- fixed ICs, 2 inputs -- just restricted to
the exponential regions.)

After training, FOR EACH exponential region we pick TWO distinct ODEs A and B, have the
net predict each solution, and form their SUM by adding the predicted coefficient
vectors.  Note the math:

  * y_A solves a 2nd-order ODE with roots {a1,a2}; y_B solves one with roots {b1,b2}.
  * y_A + y_B is NOT a 2nd-order solution -- it solves the 4th-order ODE
        (D-a1)(D-a2)(D-b1)(D-b2) y = 0,
    i.e. the sum escapes the 2nd-order family.
  * But Taylor coefficients add linearly, so we just add the two predicted c_n vectors.
    Hence the "sum" recovery is a SUPERPOSITION / LINEARITY consistency check: it is
    accurate exactly when the two individual predictions are.

We score y_A, y_B, and y_A+y_B against the analytic series.

Loss (same knobs as generalize_pinn.jl):
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
N  = 12                      # series degree: c_0 .. c_N
a0 = F(1.0); a1 = F(0.0)     # fixed initial conditions y(0)=a0, y'(0)=a1

pde_weight = F(1.0)          # ODE residual term
ic_weight  = F(1.0)          # initial-condition term
sup_weight = F(1.0)          # supervised term (0 = pure PINN; >0 blends in labels)

tau_lim   = F(2.0)
delta_lim = F(2.0)
n_total   = 800

x_left   = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))

adam_iters  = 3000
lbfgs_iters = 2000

# Only REAL-eigenvalue regions are "exponential".
exp_regions = [:saddle, :stable_node, :unstable_node]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
fact_vec = F.(factorial.(0:N))
function true_cn(tau, delta)
    a = zeros(F, N + 1); a[1] = a0; a[2] = a1
    for n in 0:(N - 2)
        a[n + 3] = tau * a[n + 2] - delta * a[n + 1]
    end
    return a ./ fact_vec
end

function region(tau, delta)
    delta < 0 && return :saddle
    delta == 0 && return :degenerate
    disc = tau^2 - 4 * delta
    tau == 0 && return :center
    disc > 0 && return tau < 0 ? :stable_node   : :unstable_node
    disc < 0 && return tau < 0 ? :stable_spiral : :unstable_spiral
    return :star
end
eigvals_td(tau, delta) = ((tau + sqrt(max(tau^2 - 4delta, 0.0))) / 2,
                          (tau - sqrt(max(tau^2 - 4delta, 0.0))) / 2)

# ---------------------------------------------------------------------------
# Dataset: sample exponential ODEs (real eigenvalues only).
# ---------------------------------------------------------------------------
taus = F[]; deltas = F[]
while length(taus) < n_total
    t = rand(F) * (2 * tau_lim)   - tau_lim
    d = rand(F) * (2 * delta_lim) - delta_lim
    region(t, d) in exp_regions && (push!(taus, t); push!(deltas, d))
end
println("sampled $n_total exponential ODEs over $exp_regions")

X = permutedims(hcat(taus, deltas))                                  # 2 × n_total
Y = reduce(hcat, [true_cn(taus[k], deltas[k]) for k in 1:n_total])   # (N+1) × n_total

perm = Random.shuffle(1:n_total); n_te = round(Int, 0.2 * n_total)
test_idx = perm[1:n_te]; train_idx = perm[n_te+1:end]
X_tr, Y_tr = X[:, train_idx], Y[:, train_idx]
X_te, Y_te = X[:, test_idx],  Y[:, test_idx]
n_tr, n_te = length(train_idx), length(test_idx)
println("train=$n_tr  test=$n_te")

# ---------------------------------------------------------------------------
# Power matrices (built once): u, u', u'' from coeffs at collocation points.
# ---------------------------------------------------------------------------
Pu  = F[ xs[m]^(i - 1)                                    for m in 1:n_colloc, i in 1:N+1 ]
Pu1 = F[ (i - 1) >= 1 ? (i - 1) * xs[m]^(i - 2) : 0       for m in 1:n_colloc, i in 1:N+1 ]
Pu2 = F[ (i - 1) >= 2 ? (i - 1)*(i - 2)*xs[m]^(i - 3) : 0 for m in 1:n_colloc, i in 1:N+1 ]

# ---------------------------------------------------------------------------
# Network: (tau, Delta) -> c_n   (same architecture as generalize_pinn.jl)
# ---------------------------------------------------------------------------
net = Lux.Chain(Lux.Dense(2, 64, tanh), Lux.Dense(64, 64, tanh),
                Lux.Dense(64, 64, tanh), Lux.Dense(64, N + 1))
p_init, st = Lux.setup(Random.default_rng(), net)
p_init_ca  = ComponentArray(p_init)

function pinn_loss(p, Xb, Yb, nb)
    C  = first(net(Xb, p, st))
    τ  = Xb[1:1, :]; Δ = Xb[2:2, :]
    resid    = (Pu2 * C) .- τ .* (Pu1 * C) .+ Δ .* (Pu * C)
    loss_pde = sum(abs2, resid) / (n_colloc * nb)
    loss_ic  = (sum(abs2, C[1, :] .- a0) + sum(abs2, C[2, :] .- a1)) / nb
    loss_sup = sup_weight == 0 ? zero(F) : sum(abs2, C .- Yb) / ((N + 1) * nb)
    return pde_weight * loss_pde + ic_weight * loss_ic + sup_weight * loss_sup
end
loss_fn(p, _) = pinn_loss(p, X_tr, Y_tr, n_tr)

function rel_rmse(p, Xb, Yb, nb)
    C = first(net(Xb, p, st))
    diff2 = sum(abs2, C .- Yb; dims = 1)
    nrm2  = sum(abs2, Yb; dims = 1) .+ F(1e-8)
    return sqrt(sum(diff2 ./ nrm2) / nb)
end

# ---------------------------------------------------------------------------
# Train (Adam -> LBFGS)
# ---------------------------------------------------------------------------
train_hist = Float64[]; test_hist = Float64[]
p_bar = Progress(adam_iters, desc = "exp-family PINN Adam ")
ic = 0
callback = function (state, l)
    global ic += 1
    push!(train_hist, rel_rmse(state.u, X_tr, Y_tr, n_tr))
    push!(test_hist,  rel_rmse(state.u, X_te, Y_te, n_te))
    ProgressMeter.next!(p_bar; showvalues = [(:iter, ic), (:loss, l),
                        (:train_relerr, train_hist[end]), (:test_relerr, test_hist[end])])
    return false
end
prob = OptimizationProblem(OptimizationFunction(loss_fn, Optimization.AutoZygote()), p_init_ca)
res  = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); callback = callback, maxiters = adam_iters)

n_adam = length(train_hist)
p_bar2 = Progress(lbfgs_iters, desc = "exp-family PINN LBFGS ")
i2 = 0
cb2 = function (state, l)
    global i2 += 1
    push!(train_hist, rel_rmse(state.u, X_tr, Y_tr, n_tr))
    push!(test_hist,  rel_rmse(state.u, X_te, Y_te, n_te))
    ProgressMeter.next!(p_bar2; showvalues = [(:iter, i2), (:loss, l),
                        (:train_relerr, train_hist[end]), (:test_relerr, test_hist[end])])
    return false
end
res = solve(remake(prob; u0 = res.u), OptimizationOptimJL.LBFGS(); callback = cb2, maxiters = lbfgs_iters)
p_trained = res.u

println("\nweights: pde=$pde_weight ic=$ic_weight sup=$sup_weight")
println("Final train rel-rmse = ", round(train_hist[end]; sigdigits = 4))
println("Final test  rel-rmse = ", round(test_hist[end];  sigdigits = 4))

plot_loss = plot(1:length(train_hist), max.(train_hist, 1e-20), yscale = :log10,
                 label = "train", lw = 2, title = "RMS rel error vs Iteration (exp-family PINN)",
                 xlabel = "Iteration (Adam then LBFGS)", ylabel = "rel-RMSE")
plot!(plot_loss, 1:length(test_hist), max.(test_hist, 1e-20), label = "test", lw = 2)
vline!(plot_loss, [n_adam + 0.5], ls = :dash, color = :gray, label = "Adam | LBFGS")
savefig(plot_loss, "data/exp_family_loss.png")

# ---------------------------------------------------------------------------
# Three tests of the trained net (training structure unchanged -- only the query):
#   (1) MEMORIZATION : an ODE taken directly FROM the training set.
#   (2) GENERALIZATION: two ODEs OUTSIDE the training set. We place them just past
#       the sampling box (|τ| > tau_lim) but keep them real-eigenvalue, so they stay
#       "exponential" -- a clean extrapolation test.
#   (3) SUM           : y_A + y_B, recovered by adding the two predicted coeff vectors.
#                       (y_A+y_B is a 4th-order solution; coeffs add linearly.)
# ---------------------------------------------------------------------------
xfine = collect(range(x_left, x_right, length = 200))
eval_series(c) = [sum(c[i] * x^(i - 1) for i in 1:N+1) for x in xfine]
relerr(pred, truth) = sqrt(sum(abs2, pred .- truth) / (sum(abs2, truth) + 1e-12))
predict(τ, Δ) = vec(first(net(reshape(F[τ, Δ], 2, 1), p_trained, st)))

# (1) memorization: reuse an actual training column
k_mem  = train_idx[1]
τm, Δm = taus[k_mem], deltas[k_mem]
c_mem  = predict(τm, Δm); a_mem = true_cn(τm, Δm); e_mem = relerr(c_mem, a_mem)

# (2) two ODEs outside the training box (|τ| > tau_lim), both real-eigenvalue
(τA, ΔA) = (F(2.5),  F(-1.0))    # saddle,      just past +τ-limit
(τB, ΔB) = (F(-2.5), F(1.5))     # stable node, just past -τ-limit
cA = predict(τA, ΔA); aA = true_cn(τA, ΔA); eA = relerr(cA, aA)
cB = predict(τB, ΔB); aB = true_cn(τB, ΔB); eB = relerr(cB, aB)

# (3) sum of the two out-of-training ODEs
cS = cA .+ cB; aS = aA .+ aB; eS = relerr(cS, aS)

println("\n--- coefficient rel-error vs analytic ---")
println(rpad("test", 26), rpad("(τ,Δ)", 16), rpad("region", 16), "rel-err")
println(rpad("(1) memorize  in-train", 26), rpad("($(round(τm;digits=2)),$(round(Δm;digits=2)))", 16),
        rpad(string(region(τm, Δm)), 16), round(e_mem; sigdigits = 3))
println(rpad("(2) ODE A     out-train", 26), rpad("($τA,$ΔA)", 16),
        rpad(string(region(τA, ΔA)), 16), round(eA; sigdigits = 3))
println(rpad("(2) ODE B     out-train", 26), rpad("($τB,$ΔB)", 16),
        rpad(string(region(τB, ΔB)), 16), round(eB; sigdigits = 3))
println(rpad("(3) sum A+B   out-train", 26), rpad("--", 16), rpad("4th-order", 16),
        round(eS; sigdigits = 3))

# --- plots: memorization panel + out-of-training (A, B, A+B) panel ---
pl_mem = plot(title = "memorization (in-train)  (τ,Δ)=($(round(τm;digits=2)),$(round(Δm;digits=2)))",
              xlabel = "x", ylabel = "u(x)", legend = :topleft)
plot!(pl_mem, xfine, eval_series(c_mem), lw = 2, color = 1, label = "pred")
plot!(pl_mem, xfine, eval_series(a_mem), lw = 1, ls = :dash, color = 1, label = "true")

pl_gen = plot(title = "out-of-training:  A, B, and A+B", xlabel = "x", ylabel = "u(x)", legend = :topleft)
plot!(pl_gen, xfine, eval_series(cA), lw = 2, color = 1, label = "A pred")
plot!(pl_gen, xfine, eval_series(aA), lw = 1, ls = :dash, color = 1, label = "A true")
plot!(pl_gen, xfine, eval_series(cB), lw = 2, color = 2, label = "B pred")
plot!(pl_gen, xfine, eval_series(aB), lw = 1, ls = :dash, color = 2, label = "B true")
plot!(pl_gen, xfine, eval_series(cS), lw = 2, color = 3, label = "A+B pred")
plot!(pl_gen, xfine, eval_series(aS), lw = 1, ls = :dash, color = 3, label = "A+B true")

test_plot = plot(pl_mem, pl_gen; layout = (1, 2), size = (1200, 460))
savefig(test_plot, "data/exp_family_sum.png")

println("\nPlots: data/exp_family_loss.png , data/exp_family_sum.png")
