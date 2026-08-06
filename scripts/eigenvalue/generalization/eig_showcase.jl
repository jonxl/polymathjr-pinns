#=
SHOWCASE -- power-series vs eig, solution-curve plots (not heatmaps) for every test type.

Trains BOTH a power-series net and a unified-eig net on the SAME exponential family (saddle
+ stable/unstable node, in-box), then overlays them against the analytic truth on each kind
of test (curves) plus a grouped bar chart of solution-space rel-L2.

Test types:
  in-distribution :  memorize (in-train ODE) , in-family (held-out, same dist)
  out-of-range    :  ODE A (saddle, |τ|=2.5) , ODE B (node, |τ|=2.5)
  out-of-family   :  spiral , center   (oscillatory regions NOT in training)

Uses the unified form u = e^{μx}(A C(k,x) + B S(k,x)), k=τ²/4-Δ, so the out-of-family
regions are REPRESENTABLE -- any failure there is genuine generalization, not the ansatz.
=#

using Lux
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote, ComponentArrays
using Plots
import Random
using Random: MersenneTwister
using LinearAlgebra

isdir("data") || mkpath("data")
F = Float32
Random.seed!(1234)

a0 = F(1.0); a1 = F(0.0)
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)
tau_lim = F(2.0); delta_lim = F(2.0)
n_total = 800
x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))
xfine = collect(range(x_left, x_right, length = 200))
adam_iters  = 3000
lbfgs_iters = 2000
exp_regions = [:saddle, :stable_node, :unstable_node]

function region(tau, delta)
    delta < 0 && return :saddle
    delta == 0 && return :degenerate
    disc = tau^2 - 4 * delta
    tau == 0 && return :center
    disc > 0 && return tau < 0 ? :stable_node   : :unstable_node
    disc < 0 && return tau < 0 ? :stable_spiral : :unstable_spiral
    return :star
end

# unified C(k,x)=cosh(√k x), S(k,x)=sinh(√k x)/√k  (series form for AD; closed form for truth)
const Pterm = 14
cC = F.([1 / factorial(big(2n))     for n in 0:Pterm])
cS = F.([1 / factorial(big(2n + 1)) for n in 0:Pterm])
xpowE = [F.(xs .^ (2n))     for n in 0:Pterm]
xpowO = [F.(xs .^ (2n + 1)) for n in 0:Pterm]
CS_series(k) = (sum(cC[n+1] .* (k .^ n) .* xpowE[n+1] for n in 0:Pterm),
                sum(cS[n+1] .* (k .^ n) .* xpowO[n+1] for n in 0:Pterm))
Cfun(k, x) = k >= 0 ? cosh(sqrt(k) * x) : cos(sqrt(-k) * x)
Sfun(k, x) = abs(k) < 1e-12 ? x : (k > 0 ? sinh(sqrt(k) * x) / sqrt(k) : sin(sqrt(-k) * x) / sqrt(-k))
function utrue_vals(τ, Δ, xpts)
    μ = Float64(τ) / 2; k = Float64(τ)^2 / 4 - Float64(Δ)
    A = Float64(a0); B = Float64(a1) - μ * Float64(a0)
    return F[ exp(μ * Float64(x)) * (A * Cfun(k, Float64(x)) + B * Sfun(k, Float64(x))) for x in xpts ]
end
relerr(u, ut) = sqrt(sum(abs2, u .- ut) / (sum(abs2, ut) + 1e-12))

# ---------------------------------------------------------------------------
# Train on the exponential family
# ---------------------------------------------------------------------------
taus = F[]; deltas = F[]
while length(taus) < n_total
    t = rand(F) * (2 * tau_lim)   - tau_lim
    d = rand(F) * (2 * delta_lim) - delta_lim
    region(t, d) in exp_regions && (push!(taus, t); push!(deltas, d))
end
X = permutedims(hcat(taus, deltas))
Uall = reduce(hcat, [utrue_vals(taus[k], deltas[k], xs) for k in 1:n_total])
perm = Random.shuffle(1:n_total); n_te = round(Int, 0.2 * n_total)
test_idx = perm[1:n_te]; train_idx = perm[n_te+1:end]
X_tr = X[:, train_idx]; U_tr = Uall[:, train_idx]; n_tr = length(train_idx)

net = Lux.Chain(Lux.Dense(2, 64, tanh), Lux.Dense(64, 64, tanh),
                Lux.Dense(64, 64, tanh), Lux.Dense(64, 4))
p_init, st = Lux.setup(Random.default_rng(), net)
p_init_ca = ComponentArray(p_init)

function pinn_loss(p, Xb, Ub, nb)
    O = first(net(Xb, p, st)); μ = O[1:1, :]; k = O[2:2, :]; A = O[3:3, :]; B = O[4:4, :]
    τ = Xb[1:1, :]; Δ = Xb[2:2, :]
    C, S = CS_series(k); v = A .* C .+ B .* S; vp = A .* (k .* S) .+ B .* C
    E = exp.(xs * μ); U = E .* v
    resid = E .* ((μ .^ 2 .+ k .- τ .* μ .+ Δ) .* v .+ (2 .* μ .- τ) .* vp)
    lp = sum(abs2, resid) / (n_colloc * nb)
    li = (sum(abs2, A .- a0) + sum(abs2, (μ .* A .+ B) .- a1)) / nb
    ls = sup_weight == 0 ? zero(F) : sum(abs2, U .- Ub) / (n_colloc * nb)
    return pde_weight * lp + ic_weight * li + sup_weight * ls
end
loss_fn(p, _) = pinn_loss(p, X_tr, U_tr, n_tr)

prob = OptimizationProblem(OptimizationFunction(loss_fn, Optimization.AutoZygote()), p_init_ca)
res  = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); maxiters = adam_iters)
res  = solve(remake(prob; u0 = res.u), OptimizationOptimJL.LBFGS(); maxiters = lbfgs_iters)
p_trained = res.u
println("trained on exponential family (n_tr=$n_tr)")

# ---------------------------------------------------------------------------
# Also train a POWER-SERIES model on the SAME data, for comparison.
#   net (tau,Delta) -> [c_0..c_N],  u(x)=sum c_n x^n
# ---------------------------------------------------------------------------
N = 12
fact_vec = F.([factorial(big(k)) for k in 0:N])
function true_cn(τ, Δ)
    a = zeros(F, N + 1); a[1] = a0; a[2] = a1
    for n in 0:(N - 2); a[n + 3] = τ * a[n + 2] - Δ * a[n + 1]; end
    return a ./ fact_vec
end
Pu  = F[ xs[m]^(i - 1)                                    for m in 1:n_colloc, i in 1:N+1 ]
Pu1 = F[ (i - 1) >= 1 ? (i - 1) * xs[m]^(i - 2) : 0       for m in 1:n_colloc, i in 1:N+1 ]
Pu2 = F[ (i - 1) >= 2 ? (i - 1)*(i - 2)*xs[m]^(i - 3) : 0 for m in 1:n_colloc, i in 1:N+1 ]
Y_tr_ps = reduce(hcat, [true_cn(taus[k], deltas[k]) for k in train_idx])
net_ps = Lux.Chain(Lux.Dense(2, 64, tanh), Lux.Dense(64, 64, tanh),
                   Lux.Dense(64, 64, tanh), Lux.Dense(64, N + 1))
p0_ps, st_ps = Lux.setup(Random.default_rng(), net_ps)
function loss_ps(p, _)
    C = first(net_ps(X_tr, p, st_ps)); τ = X_tr[1:1, :]; Δ = X_tr[2:2, :]
    lp = sum(abs2, (Pu2 * C) .- τ .* (Pu1 * C) .+ Δ .* (Pu * C)) / (n_colloc * n_tr)
    li = (sum(abs2, C[1, :] .- a0) + sum(abs2, C[2, :] .- a1)) / n_tr
    ls = sum(abs2, C .- Y_tr_ps) / ((N + 1) * n_tr)
    return pde_weight * lp + ic_weight * li + sup_weight * ls
end
prob_ps = OptimizationProblem(OptimizationFunction(loss_ps, Optimization.AutoZygote()), ComponentArray(p0_ps))
rps = solve(prob_ps, OptimizationOptimisers.Adam(F(1e-3)); maxiters = adam_iters)
rps = solve(remake(prob_ps; u0 = rps.u), OptimizationOptimJL.LBFGS(); maxiters = lbfgs_iters)
p_ps = rps.u
u_ps(τ, Δ, xpts) = (c = vec(first(net_ps(reshape(F[τ, Δ], 2, 1), p_ps, st_ps)));
                    [sum(c[i] * x^(i - 1) for i in 1:N+1) for x in xpts])
println("power-series model trained")

# ---------------------------------------------------------------------------
# Prediction (closed form, exact) + per-scenario plotting
# ---------------------------------------------------------------------------
function predict(τ, Δ)
    O = vec(first(net(reshape(F[τ, Δ], 2, 1), p_trained, st)))
    return O[1], O[2], O[3], O[4]            # μ, k, A, B
end
function u_pred_grid(τ, Δ, xpts)
    μ, k, A, B = predict(τ, Δ)
    return [exp(μ * x) * (A * Cfun(k, x) + B * Sfun(k, x)) for x in xpts]
end

km = train_idx[1]; kf = test_idx[1]
scenarios = [
    ("memorize (in-train)",   taus[km],   deltas[km]),
    ("in-family (held-out)",  taus[kf],   deltas[kf]),
    ("out-of-range A (saddle)", F(2.5),  F(-1.0)),
    ("out-of-range B (node)",   F(-2.5), F(1.5)),
    ("out-of-family (spiral)",  F(-1.0), F(1.0)),
    ("out-of-family (center)",  F(0.0),  F(1.0)),
]

panels = Plots.Plot[]; errs_eig = Float64[]; errs_ps = Float64[]
println("\n", rpad("scenario", 26), rpad("power-series", 16), "eig")
for (nm, τ, Δ) in scenarios
    ut   = utrue_vals(τ, Δ, xfine)
    ueig = u_pred_grid(τ, Δ, xfine); ups = u_ps(τ, Δ, xfine)
    e_eig = relerr(ueig, ut); e_ps = relerr(ups, ut)
    push!(errs_eig, e_eig); push!(errs_ps, e_ps)
    pl = plot(xfine, ut, lw = 3, color = :black, ls = :dash, label = "true",
              title = "$nm\nPS=$(round(e_ps;sigdigits=2))  EIG=$(round(e_eig;sigdigits=2))",
              xlabel = "x", ylabel = "u(x)", legend = :best, titlefontsize = 8)
    plot!(pl, xfine, ups,  lw = 2, color = :orange,     label = "power series")
    plot!(pl, xfine, ueig, lw = 2, color = :dodgerblue, label = "eig")
    push!(panels, pl)
    println(rpad(nm, 26), rpad(round(e_ps; sigdigits = 3), 16), round(e_eig; sigdigits = 3))
end

fig = plot(panels...; layout = (2, 3), size = (1450, 820))
savefig(fig, "data/eig_showcase_curves.png")

# ---------------------------------------------------------------------------
# Grouped bar chart (replaces the heatmap): power-series vs eig, by test type
# ---------------------------------------------------------------------------
bar_names = ["memorize", "in-family", "range A", "range B", "out-fam\nspiral", "out-fam\ncenter"]
xp = collect(1:length(scenarios))
bp = bar(xp .- 0.2, max.(errs_ps, 1e-6); bar_width = 0.4, label = "power series", color = :orange,
         yscale = :log10, ylabel = "solution rel-L2 (log)", legend = :topleft,
         title = "power-series vs eig, by test type", xticks = (xp, bar_names), size = (950, 560))
bar!(bp, xp .+ 0.2, max.(errs_eig, 1e-6); bar_width = 0.4, label = "eig", color = :dodgerblue)
savefig(bp, "data/eig_showcase_bars.png")

println("\nPlots: data/eig_showcase_curves.png , data/eig_showcase_bars.png")
