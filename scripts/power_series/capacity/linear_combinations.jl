#=
LINEAR COMBINATIONS of one ODE, at FIXED c1+c2  (the step after single-ODE memorization).

Fix ONE ODE  y'' - τ0 y' + Δ0 y = 0  with roots λ1, λ2, and hold the amplitude sum
constant: c1 + c2 = S.  Then EVERY solution shares the same initial value u(0)=S, and the
family is 1-parameter (parametrized by c1, with c2 = S - c1) -- only the *combination*
(and hence u'(0)=λ1 c1+λ2 c2) varies.  We train the net to map (c1,c2) on that line to the
solution and test on c1 NOT seen in training -- interpolation (|c1|≤c_lim) and
extrapolation (outside).

    net : (c1, c2) -> [a0 .. a_N]     (power-series coeffs, u(x) = Σ aₙ xⁿ)

Because a0 = c1+c2 = S is now CONSTANT, u(0) is trivially reproduced everywhere (a constant
extrapolates), so the extrapolation failure shows up in the SHAPE/slope (a1 = 2c1 - S is
linear in c1 and won't extrapolate), not in the starting point.

Loss: loss_pde (ODE residual) + loss_ic (a0->c1+c2, a1->λ1 c1+λ2 c2) + loss_sup (true coeffs).
Metric: solution-space relative L2.
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

S = F(1.0)                                       # FIXED amplitude sum: c1 + c2 = S (so u(0)=S for all)
N = 12
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)
n_colloc = 50; xs = collect(range(F(0.0), F(1.0), length = n_colloc))
xfine = collect(range(F(0.0), F(1.0), length = 200)); Mf = length(xfine)
adam_iters = 20000; lbfgs_iters = 50000           # extensive: make training a non-issue

c_lim = F(1.0)                                    # training: c1 ∈ [-c_lim, c_lim]  (c2 = S - c1)
n_train = 600
Lc = 3.0; Ng = 241; eps_tol = 0.10                # test-sweep half-width (in c1), resolution, tolerance

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

# amplitudes on the line c1+c2=S: given c1, c2 = S - c1
line(c1v) = permutedims(hcat(c1v, S .- c1v))      # 2×n  rows (c1, c2)

# ---------------------------------------------------------------------------
# Training data: c1 in the box, c2 = S - c1; targets are the (linear) true coeffs
# ---------------------------------------------------------------------------
c1_tr = F.(rand(n_train) .* (2c_lim) .- c_lim)
Xtr = line(c1_tr)                                             # 2×n_train  (c1, S-c1)
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
println("fixed ODE (τ,Δ)=($τ0,$Δ0), roots λ=($λ1,$λ2), fixed sum c1+c2=$S")
println("trained on c1 ∈ [-$c_lim,$c_lim]  (c2 = $S - c1)  n=$n_train")

# ---------------------------------------------------------------------------
# Error vs c1 sweep along the line (all share u(0)=S) + interpolation vs extrapolation
# ---------------------------------------------------------------------------
c1g = collect(range(-Lc, Lc, length = Ng))
Xg = line(c1g)
Cg = first(net(Xg, p_tr, st))
Upred = Pufine * Cg; Utrue = e1 * Xg[1:1, :] .+ e2 * Xg[2:2, :]
errs = vec(sqrt.(sum(abs2, Upred .- Utrue; dims = 1) ./ (sum(abs2, Utrue; dims = 1) .+ F(1e-12))))
in_range = abs.(c1g) .<= c_lim
println("median rel-L2  interpolation (|c1|≤$c_lim) = ", round(median(errs[in_range]); sigdigits = 3),
        "   extrapolation = ", round(median(errs[.!in_range]); sigdigits = 3))

pe = plot(c1g, max.(errs, 1e-6); yscale = :log10, lw = 2, legend = false, color = :purple,
          xlabel = "c₁   (c₂ = $S − c₁)", ylabel = "solution rel-L2 (log)",
          title = "error vs c₁ along fixed sum c₁+c₂=$S, ε=$eps_tol")
vspan!(pe, [-Float64(c_lim), Float64(c_lim)]; color = :green, alpha = 0.12, label = "")
hline!(pe, [eps_tol]; color = :red, ls = :dash, label = "")
savefig(pe, "data/linear_combinations_sweep.png")

# ---------------------------------------------------------------------------
# Sample solution curves (all start at u(0)=S): memorize, in-range, two extrapolated
# ---------------------------------------------------------------------------
u_pred(c1) = Pufine * vec(first(net(reshape(F[c1, S - c1], 2, 1), p_tr, st)))
samples = [(Float64(c1_tr[1]), "memorization (trained)"),
           (0.3, "in-range (held-out)"),
           (2.5, "extrapolated"),
           (-1.5, "extrapolated")]
panels = Plots.Plot[]
for (c1, tag) in samples
    c2 = Float64(S) - c1
    up = u_pred(F(c1)); ut = c1 .* e1 .+ c2 .* e2
    e = sqrt(sum(abs2, up .- ut) / (sum(abs2, ut) + 1e-12))
    pl = plot(xfine, ut, lw = 3, color = :black, ls = :dash, label = "true",
              title = "c₁=$(round(c1;digits=2)), c₂=$(round(c2;digits=2))  $tag\nrel-L2=$(round(e;sigdigits=2))",
              xlabel = "x", ylabel = "u(x)", legend = :best, titlefontsize = 8)
    plot!(pl, xfine, up, lw = 2, color = :dodgerblue, label = "pred")
    push!(panels, pl)
end
savefig(plot(panels...; layout = (2, 2), size = (1000, 720)), "data/linear_combinations_curves.png")

println("\nPlots: data/linear_combinations_sweep.png , data/linear_combinations_curves.png")
