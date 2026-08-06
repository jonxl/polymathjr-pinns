#=
EIGENVALUE / EXPONENTIAL representation  (NEW data construction; power-series scripts untouched).

Instead of outputting power-series coefficients, the network outputs the solution's
EXPONENTIAL parameters directly:

    net : (tau, Delta) -> (lambda1, lambda2, c1, c2)
    u(x) = c1 * exp(lambda1 * x) + c2 * exp(lambda2 * x)      ("hand-written" form)

No power series, no truncation, no high-order differentiation matrices -- the things that
caused every monomial / Chebyshev conditioning failure.

ALTERED LOSS (the key change vs the power-series scripts):
  loss_pde : PDE residual computed ANALYTICALLY from the exponential form
        u'' - tau u' + Delta u = sum_k c_k (lambda_k^2 - tau lambda_k + Delta) e^{lambda_k x}
     -> minimizing it drives each lambda_k to a ROOT of r^2 - tau r + Delta = 0.
        (No u'' differentiation matrix, so it is well conditioned and not hypersensitive.)
  loss_ic  : u(0)=c1+c2 -> a0 ,  u'(0)=lambda1 c1 + lambda2 c2 -> a1
        (also breaks the lambda1=lambda2 degeneracy)
  loss_sup : SOLUTION-space MSE vs the analytic true u at the collocation points.

Metric reported: SOLUTION-space relative L2 (basis-independent), same as we settled on.

Restricted to the REAL-eigenvalue (exponential) regions: saddle / stable node / unstable
node. Spirals & centers (complex roots) need a different ansatz.

CAVEAT: this BAKES IN the two-exponential ansatz, so it tests "can the net learn the
characteristic roots" -- an easier, different question than the generic power-series PINN.
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
a0 = F(1.0); a1 = F(0.0)
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)
tau_lim = F(2.0); delta_lim = F(2.0)
n_total = 800
x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))      # n_colloc-vector (column)
xfine = collect(range(x_left, x_right, length = 200))
adam_iters  = 3000
lbfgs_iters = 2000
exp_regions = [:saddle, :stable_node, :unstable_node]

testA = (F(2.5),  F(-1.0))     # out-of-box saddle
testB = (F(-2.5), F(1.5))      # out-of-box stable node

function region(tau, delta)
    delta < 0 && return :saddle
    delta == 0 && return :degenerate
    disc = tau^2 - 4 * delta
    tau == 0 && return :center
    disc > 0 && return tau < 0 ? :stable_node   : :unstable_node
    disc < 0 && return tau < 0 ? :stable_spiral : :unstable_spiral
    return :star
end

# analytic solution c1 e^{l1 x} + c2 e^{l2 x} with ICs (a0,a1); real roots in exp regions
function utrue_vals(τ, Δ, xpts)
    s  = sqrt(Float64(τ)^2 - 4 * Float64(Δ))
    l1 = (Float64(τ) + s) / 2; l2 = (Float64(τ) - s) / 2
    d1 = (Float64(a1) - l2 * Float64(a0)) / (l1 - l2); d2 = Float64(a0) - d1
    return F[ d1 * exp(l1 * Float64(x)) + d2 * exp(l2 * Float64(x)) for x in xpts ]
end
true_roots(τ, Δ) = ((Float64(τ) + sqrt(Float64(τ)^2 - 4Float64(Δ))) / 2,
                    (Float64(τ) - sqrt(Float64(τ)^2 - 4Float64(Δ))) / 2)
relerr(u, ut) = sqrt(sum(abs2, u .- ut) / (sum(abs2, ut) + 1e-12))

# ---------------------------------------------------------------------------
# Dataset (exponential regions only); supervised targets are SOLUTION values.
# ---------------------------------------------------------------------------
taus = F[]; deltas = F[]
while length(taus) < n_total
    t = rand(F) * (2 * tau_lim)   - tau_lim
    d = rand(F) * (2 * delta_lim) - delta_lim
    region(t, d) in exp_regions && (push!(taus, t); push!(deltas, d))
end
X = permutedims(hcat(taus, deltas))                                   # 2 × n_total
Uall = reduce(hcat, [utrue_vals(taus[k], deltas[k], xs) for k in 1:n_total])   # n_colloc × n_total

perm = Random.shuffle(1:n_total); n_te = round(Int, 0.2 * n_total)
test_idx = perm[1:n_te]; train_idx = perm[n_te+1:end]
X_tr = X[:, train_idx]; U_tr = Uall[:, train_idx]; n_tr = length(train_idx)
println("sampled $n_total exponential ODEs; train=$n_tr test=$n_te")

# ---------------------------------------------------------------------------
# Network: (tau, Delta) -> (lambda1, lambda2, c1, c2)
# ---------------------------------------------------------------------------
net = Lux.Chain(Lux.Dense(2, 64, tanh), Lux.Dense(64, 64, tanh),
                Lux.Dense(64, 64, tanh), Lux.Dense(64, 4))
p_init, st = Lux.setup(Random.default_rng(), net)
p_init_ca  = ComponentArray(p_init)

# reconstruct u and the residual at collocation points from the 4 params (all 1×nb)
function pinn_loss(p, Xb, Ub, nb)
    O  = first(net(Xb, p, st))                  # 4 × nb
    λ1 = O[1:1, :]; λ2 = O[2:2, :]; c1 = O[3:3, :]; c2 = O[4:4, :]
    τ  = Xb[1:1, :]; Δ = Xb[2:2, :]

    E1 = exp.(xs * λ1); E2 = exp.(xs * λ2)       # n_colloc × nb
    U  = c1 .* E1 .+ c2 .* E2                     # u at collocation points

    r1 = λ1 .^ 2 .- τ .* λ1 .+ Δ                  # characteristic value per mode (1×nb)
    r2 = λ2 .^ 2 .- τ .* λ2 .+ Δ
    resid = (c1 .* r1) .* E1 .+ (c2 .* r2) .* E2
    loss_pde = sum(abs2, resid) / (n_colloc * nb)

    loss_ic  = (sum(abs2, (c1 .+ c2) .- a0) +
                sum(abs2, (λ1 .* c1 .+ λ2 .* c2) .- a1)) / nb
    loss_sup = sup_weight == 0 ? zero(F) : sum(abs2, U .- Ub) / (n_colloc * nb)
    return pde_weight * loss_pde + ic_weight * loss_ic + sup_weight * loss_sup
end
loss_fn(p, _) = pinn_loss(p, X_tr, U_tr, n_tr)

# raw component breakdown (unweighted) for any batch
function loss_components(p, Xb, Ub, nb)
    O  = first(net(Xb, p, st)); λ1 = O[1:1, :]; λ2 = O[2:2, :]; c1 = O[3:3, :]; c2 = O[4:4, :]
    τ  = Xb[1:1, :]; Δ = Xb[2:2, :]
    E1 = exp.(xs * λ1); E2 = exp.(xs * λ2); U = c1 .* E1 .+ c2 .* E2
    resid = (c1 .* (λ1 .^ 2 .- τ .* λ1 .+ Δ)) .* E1 .+ (c2 .* (λ2 .^ 2 .- τ .* λ2 .+ Δ)) .* E2
    lp = sum(abs2, resid) / (n_colloc * nb)
    li = (sum(abs2, (c1 .+ c2) .- a0) + sum(abs2, (λ1 .* c1 .+ λ2 .* c2) .- a1)) / nb
    ls = sum(abs2, U .- Ub) / (n_colloc * nb)
    return lp, li, ls
end

# ---------------------------------------------------------------------------
# Train (Adam -> LBFGS)
# ---------------------------------------------------------------------------
prob = OptimizationProblem(OptimizationFunction(loss_fn, Optimization.AutoZygote()), p_init_ca)
i1 = 0
cb1 = (s, l) -> (global i1 += 1; i1 % 500 == 0 && println("  adam $i1  loss=$(round(l;sigdigits=4))"); false)
res  = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); callback = cb1, maxiters = adam_iters)
res  = solve(remake(prob; u0 = res.u), OptimizationOptimJL.LBFGS(); maxiters = lbfgs_iters)
p_trained = res.u

lp, li, ls = loss_components(p_trained, X_tr, U_tr, n_tr)
println("\nfinal train loss components (raw):  pde=$(round(lp;sigdigits=3))  ",
        "ic=$(round(li;sigdigits=3))  sup=$(round(ls;sigdigits=3))")

# ---------------------------------------------------------------------------
# Evaluate (solution-space rel-L2). Also show recovered vs true eigenvalues.
# ---------------------------------------------------------------------------
function predict(τ, Δ)
    O = vec(first(net(reshape(F[τ, Δ], 2, 1), p_trained, st)))
    return O[1], O[2], O[3], O[4]                  # λ1, λ2, c1, c2
end
u_pred(τ, Δ, xpts) = (p = predict(τ, Δ); [p[3] * exp(p[1] * x) + p[4] * exp(p[2] * x) for x in xpts])

km = train_idx[1]
e_mem = relerr(u_pred(taus[km], deltas[km], xfine), utrue_vals(taus[km], deltas[km], xfine))
uA = u_pred(testA..., xfine); eA = relerr(uA, utrue_vals(testA..., xfine))
uB = u_pred(testB..., xfine); eB = relerr(uB, utrue_vals(testB..., xfine))
eS = relerr(uA .+ uB, utrue_vals(testA..., xfine) .+ utrue_vals(testB..., xfine))

println("\n--- solution-space rel-L2 ---")
println(rpad("test", 26), rpad("(τ,Δ)", 16), "rel-L2")
println(rpad("memorize (in-train)", 26), rpad("($(round(taus[km];digits=2)),$(round(deltas[km];digits=2)))", 16), round(e_mem; sigdigits = 3))
println(rpad("ODE A (out-of-box)", 26), rpad("$(testA)", 16), round(eA; sigdigits = 3))
println(rpad("ODE B (out-of-box)", 26), rpad("$(testB)", 16), round(eB; sigdigits = 3))
println(rpad("sum A+B", 26), rpad("--", 16), round(eS; sigdigits = 3))

println("\n--- recovered vs true eigenvalues ---")
for (nm, td) in [("ODE A", testA), ("ODE B", testB)]
    p = predict(td...); tr = true_roots(td...)
    println("$nm  pred λ = ($(round(p[1];digits=3)), $(round(p[2];digits=3)))   ",
            "true λ = ($(round(tr[1];digits=3)), $(round(tr[2];digits=3)))")
end

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
pl = plot(title = "eig representation: out-of-box A, B, sum", xlabel = "x", ylabel = "u(x)", legend = :topleft)
plot!(pl, xfine, uA, lw = 2, color = 1, label = "A pred")
plot!(pl, xfine, utrue_vals(testA..., xfine), lw = 1, ls = :dash, color = 1, label = "A true")
plot!(pl, xfine, uB, lw = 2, color = 2, label = "B pred")
plot!(pl, xfine, utrue_vals(testB..., xfine), lw = 1, ls = :dash, color = 2, label = "B true")
plot!(pl, xfine, uA .+ uB, lw = 2, color = 3, label = "A+B pred")
plot!(pl, xfine, utrue_vals(testA..., xfine) .+ utrue_vals(testB..., xfine), lw = 1, ls = :dash, color = 3, label = "A+B true")
savefig(pl, "data/eig_family.png")
println("\nPlot: data/eig_family.png")
