#=
LINEAR COMBINATIONS of one ODE  (the step after single-ODE memorization).

Fix ONE ODE  y'' - τ0 y' + Δ0 y = 0  with roots λ1, λ2.  Its solution space is 2-D:
        u(x) = c1 e^{λ1 x} + c2 e^{λ2 x}.
We train a network to map the AMPLITUDES (c1, c2) -> the solution, and test on (c1, c2)
NOT seen in training -- both interpolation (inside the training box) and extrapolation
(outside it).  Because the true map is exactly LINEAR in (c1, c2), this asks whether a
tanh MLP discovers superposition, and how far it extrapolates the linear structure.

    net : (c1, c2) -> [a0 .. a_N]     (power-series coeffs, u(x) = Σ aₙ xⁿ)

Loss:
    loss_pde : ODE residual (same fixed ODE for every (c1,c2))
    loss_ic  : a0 -> c1+c2 ,  a1 -> λ1 c1 + λ2 c2      (input-dependent ICs)
    loss_sup : coeffs vs true  aₙ = (c1 λ1ⁿ + c2 λ2ⁿ)/n!   (linear in c1,c2)

Metric: solution-space relative L2.  Output: an error map over the (c1,c2) plane with the
eps-contour and the training box, plus a few sample solution curves.
=#

using Lux
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote, ComponentArrays
using Plots
import Random
using Statistics: median
using LinearAlgebra

isdir("data") || mkpath("data")
F = Float32
Random.seed!(1234)

# --- the fixed ODE ---
τ0, Δ0 = F(0.0), F(-1.0)                         # saddle: roots +1, -1  ->  u = c1 e^x + c2 e^{-x}
λ1 = (τ0 + sqrt(τ0^2 - 4Δ0)) / 2
λ2 = (τ0 - sqrt(τ0^2 - 4Δ0)) / 2

N = 12
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)
n_colloc = 50; xs = collect(range(F(0.0), F(1.0), length = n_colloc))
xfine = collect(range(F(0.0), F(1.0), length = 200)); Mf = length(xfine)
adam_iters = 20000; lbfgs_iters = 50000           # extensive: make training a non-issue

c_lim = F(1.0)                                    # training box: (c1,c2) in [-c_lim, c_lim]^2
n_train = 600
Lc = 3.0; Ng = 81; eps_tol = 0.10                # test-grid half-width, resolution, tolerance

# basis: coeffs of e^{λ x} is λ^n/n!, so true coeffs = c1*b1 + c2*b2 (linear in amplitudes)
fact_vec = F.([factorial(big(n)) for n in 0:N])
b1 = F.([λ1^n for n in 0:N]) ./ fact_vec
b2 = F.([λ2^n for n in 0:N]) ./ fact_vec

# power matrices for the residual (on xs) and reconstruction (on xfine)
Pu  = F[ xs[m]^(i - 1)                                    for m in 1:n_colloc, i in 1:N+1 ]
Pu1 = F[ (i - 1) >= 1 ? (i - 1) * xs[m]^(i - 2) : 0       for m in 1:n_colloc, i in 1:N+1 ]
Pu2 = F[ (i - 1) >= 2 ? (i - 1)*(i - 2)*xs[m]^(i - 3) : 0 for m in 1:n_colloc, i in 1:N+1 ]
Pufine = F[ xfine[m]^(i - 1) for m in 1:Mf, i in 1:N+1 ]
e1 = exp.(λ1 .* xfine); e2 = exp.(λ2 .* xfine)    # fixed basis solutions on xfine (λ fixed)
utrue_grid(Cmat) = e1 * Cmat[1:1, :] .+ e2 * Cmat[2:2, :]     # analytic u for a batch of (c1,c2)

# ---------------------------------------------------------------------------
# Training data: amplitudes in the box, targets are the (linear) true coeffs
# ---------------------------------------------------------------------------
Xtr = F.(rand(2, n_train) .* (2c_lim) .- c_lim)               # 2×n_train  (c1,c2)
Y_tr = b1 * Xtr[1:1, :] .+ b2 * Xtr[2:2, :]                   # (N+1)×n_train true coeffs

net = Lux.Chain(Lux.Dense(2, 64, tanh), Lux.Dense(64, 64, tanh), Lux.Dense(64, 64, tanh), Lux.Dense(64, N + 1))
p_init, st = Lux.setup(Random.default_rng(), net); p_init_ca = ComponentArray(p_init)

function loss_fn(p, _)
    C = first(net(Xtr, p, st)); c1 = Xtr[1:1, :]; c2 = Xtr[2:2, :]
    resid = (Pu2 * C) .- τ0 .* (Pu1 * C) .+ Δ0 .* (Pu * C)
    lp = sum(abs2, resid) / (n_colloc * n_train)
    li = (sum(abs2, C[1:1, :] .- (c1 .+ c2)) + sum(abs2, C[2:2, :] .- (λ1 .* c1 .+ λ2 .* c2))) / n_train
    ls = sum(abs2, C .- Y_tr) / ((N + 1) * n_train)
    return pde_weight * lp + ic_weight * li + sup_weight * ls
end
prob = OptimizationProblem(OptimizationFunction(loss_fn, Optimization.AutoZygote()), p_init_ca)
res = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); maxiters = adam_iters)
res = solve(remake(prob; u0 = res.u), OptimizationOptimJL.LBFGS(); maxiters = lbfgs_iters)
p_tr = res.u
println("fixed ODE (τ,Δ)=($τ0,$Δ0), roots λ=($λ1,$λ2); trained on (c1,c2)∈[-$c_lim,$c_lim]²  n=$n_train")

# ---------------------------------------------------------------------------
# Error map over the (c1,c2) plane (batched) + interpolation vs extrapolation
# ---------------------------------------------------------------------------
c1g = collect(range(-Lc, Lc, length = Ng)); c2g = collect(range(-Lc, Lc, length = Ng))
Xg = reduce(hcat, [F[a, b] for b in c2g for a in c1g])       # 2×G, c2 outer / c1 inner
Cg = first(net(Xg, p_tr, st))
Upred = Pufine * Cg; Utrue = utrue_grid(Xg)
errs = vec(sqrt.(sum(abs2, Upred .- Utrue; dims = 1) ./ (sum(abs2, Utrue; dims = 1) .+ F(1e-12))))
Emap = permutedims(reshape(errs, Ng, Ng))                    # [c2_idx, c1_idx]

in_box = [abs(a) <= c_lim && abs(b) <= c_lim for b in c2g for a in c1g]
println("median rel-L2  interpolation (in-box) = ", round(median(errs[in_box]); sigdigits = 3),
        "   extrapolation (out-of-box) = ", round(median(errs[.!in_box]); sigdigits = 3))

hm = heatmap(c1g, c2g, log10.(max.(Emap, 1e-5)); c = :viridis, colorbar_title = "log10 rel-L2",
             xlabel = "c₁", ylabel = "c₂", title = "linear-combination error over (c₁,c₂), ε=$eps_tol")
contour!(hm, c1g, c2g, Emap; levels = [eps_tol], color = :red, lw = 2, colorbar_entry = false)
plot!(hm, [-c_lim, c_lim, c_lim, -c_lim, -c_lim], [-c_lim, -c_lim, c_lim, c_lim, -c_lim];
      color = :white, ls = :dash, lw = 2, label = "train box")
savefig(hm, "data/linear_combinations_map.png")

# ---------------------------------------------------------------------------
# Sample solution curves: in-box, on the edge, and extrapolated
# ---------------------------------------------------------------------------
u_pred(c1, c2) = Pufine * vec(first(net(reshape(F[c1, c2], 2, 1), p_tr, st)))
# one actual TRAINING point (memorization), one held-out IN-BOX, two OUT-OF-BOX
samples = [(Float64(Xtr[1, 1]), Float64(Xtr[2, 1]), "memorization (trained)"),
           (0.4, -0.6, "in-box (held-out)"),
           (2.0, 2.0, "extrapolated"),
           (2.5, -1.5, "extrapolated")]
panels = Plots.Plot[]
for (c1, c2, tag) in samples
    up = u_pred(F(c1), F(c2)); ut = c1 .* e1 .+ c2 .* e2
    e = sqrt(sum(abs2, up .- ut) / (sum(abs2, ut) + 1e-12))
    pl = plot(xfine, ut, lw = 3, color = :black, ls = :dash, label = "true",
              title = "(c₁,c₂)=($(round(c1;digits=2)),$(round(c2;digits=2)))  $tag\nrel-L2=$(round(e;sigdigits=2))",
              xlabel = "x", ylabel = "u(x)", legend = :best, titlefontsize = 8)
    plot!(pl, xfine, up, lw = 2, color = :dodgerblue, label = "pred")
    push!(panels, pl)
end
savefig(plot(panels...; layout = (2, 2), size = (1000, 720)), "data/linear_combinations_curves.png")

println("\nPlots: data/linear_combinations_map.png , data/linear_combinations_curves.png")
