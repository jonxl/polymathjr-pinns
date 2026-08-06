#=
N SWEEP -- does a higher series degree reduce error on large-λ / out-of-box ODEs?

Hypothesis (from the coefficient-setup analysis): a degree-N truncated power series cannot
exactly satisfy the ODE -- the top terms leave a residual ~ a_{N+1}/(N-1)! = λ^{N+1}/(N-1)!.
For small λ this is negligible; for large λ (out-of-box ODEs, λ≈2.85) it is sizeable at
N=12 (~0.085), so the residual-minimizer is pulled away from the true Taylor coefficients.
Raising N shrinks that tail fast, so large-λ error should drop.

Same family PINN as exp_family_sum.jl (Dense(2→64→64→64→N+1), tanh, residual+ic+sup loss,
Adam→LBFGS), trained on 640 exponential ODEs. For each N we report:
    (1) memorize  -- an in-training ODE
    (2) ODE A     -- out-of-box saddle      (2.5, -1.0)
    (3) ODE B     -- out-of-box stable node (-2.5, 1.5)
    (4) sum A+B
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

# ---------------------------------------------------------------------------
# Fixed config (everything except N)
# ---------------------------------------------------------------------------
a0 = F(1.0); a1 = F(0.0)
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)
tau_lim = F(2.0); delta_lim = F(2.0)
n_total = 800
x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))
adam_iters  = 3000
lbfgs_iters = 2000
exp_regions = [:saddle, :stable_node, :unstable_node]

testA = (F(2.5),  F(-1.0))     # out-of-box saddle
testB = (F(-2.5), F(1.5))      # out-of-box stable node

Ns = [12, 16, 20, 24]

function region(tau, delta)
    delta < 0 && return :saddle
    delta == 0 && return :degenerate
    disc = tau^2 - 4 * delta
    tau == 0 && return :center
    disc > 0 && return tau < 0 ? :stable_node   : :unstable_node
    disc < 0 && return tau < 0 ? :stable_spiral : :unstable_spiral
    return :star
end
relerr(a, b) = sqrt(sum(abs2, a .- b) / (sum(abs2, b) + 1e-12))

# ---------------------------------------------------------------------------
# Train the family at a given N; return (memorize, errA, errB, errSum).
# ---------------------------------------------------------------------------
function run_for_N(N)
    fact_vec = F.([factorial(big(k)) for k in 0:N])   # big() avoids Int64 overflow at N>20
    function true_cn(tau, delta)
        a = zeros(F, N + 1); a[1] = a0; a[2] = a1
        for n in 0:(N - 2)
            a[n + 3] = tau * a[n + 2] - delta * a[n + 1]
        end
        return a ./ fact_vec
    end

    # identical ODE sample for every N (reseed before sampling)
    Random.seed!(1234)
    taus = F[]; deltas = F[]
    while length(taus) < n_total
        t = rand(F) * (2 * tau_lim)   - tau_lim
        d = rand(F) * (2 * delta_lim) - delta_lim
        region(t, d) in exp_regions && (push!(taus, t); push!(deltas, d))
    end
    X = permutedims(hcat(taus, deltas))
    Y = reduce(hcat, [true_cn(taus[k], deltas[k]) for k in 1:n_total])
    perm = Random.shuffle(1:n_total); n_te = round(Int, 0.2 * n_total)
    test_idx = perm[1:n_te]; train_idx = perm[n_te+1:end]
    X_tr, Y_tr = X[:, train_idx], Y[:, train_idx]; n_tr = length(train_idx)

    Pu  = F[ xs[m]^(i - 1)                                    for m in 1:n_colloc, i in 1:N+1 ]
    Pu1 = F[ (i - 1) >= 1 ? (i - 1) * xs[m]^(i - 2) : 0       for m in 1:n_colloc, i in 1:N+1 ]
    Pu2 = F[ (i - 1) >= 2 ? (i - 1)*(i - 2)*xs[m]^(i - 3) : 0 for m in 1:n_colloc, i in 1:N+1 ]

    net = Lux.Chain(Lux.Dense(2, 64, tanh), Lux.Dense(64, 64, tanh),
                    Lux.Dense(64, 64, tanh), Lux.Dense(64, N + 1))
    rng = MersenneTwister(1234)
    p0, st = Lux.setup(rng, net); p0ca = ComponentArray(p0)

    function loss(p, _)
        C  = first(net(X_tr, p, st))
        τ  = X_tr[1:1, :]; Δ = X_tr[2:2, :]
        resid = (Pu2 * C) .- τ .* (Pu1 * C) .+ Δ .* (Pu * C)
        lp = sum(abs2, resid) / (n_colloc * n_tr)
        li = (sum(abs2, C[1, :] .- a0) + sum(abs2, C[2, :] .- a1)) / n_tr
        ls = sum(abs2, C .- Y_tr) / ((N + 1) * n_tr)
        return pde_weight * lp + ic_weight * li + sup_weight * ls
    end
    prob = OptimizationProblem(OptimizationFunction(loss, Optimization.AutoZygote()), p0ca)
    r = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); maxiters = adam_iters)
    r = solve(remake(prob; u0 = r.u), OptimizationOptimJL.LBFGS(); maxiters = lbfgs_iters)
    p = r.u

    predict(τ, Δ) = vec(first(net(reshape(F[τ, Δ], 2, 1), p, st)))
    km = train_idx[1]
    em = relerr(predict(taus[km], deltas[km]), true_cn(taus[km], deltas[km]))
    cA = predict(testA...); aA = true_cn(testA...); eA = relerr(cA, aA)
    cB = predict(testB...); aB = true_cn(testB...); eB = relerr(cB, aB)
    eS = relerr(cA .+ cB, aA .+ aB)
    return em, eA, eB, eS
end

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
mem = Float64[]; eAs = Float64[]; eBs = Float64[]; eSs = Float64[]
for N in Ns
    println("training family at N=$N ...")
    em, eA, eB, eS = run_for_N(N)
    push!(mem, em); push!(eAs, eA); push!(eBs, eB); push!(eSs, eS)
    println("  N=$N  memorize=$(round(em;sigdigits=3))  A=$(round(eA;sigdigits=3))  ",
            "B=$(round(eB;sigdigits=3))  sum=$(round(eS;sigdigits=3))")
end

println("\nerror vs series degree N")
println(rpad("N", 6), rpad("memorize", 12), rpad("ODE A", 12), rpad("ODE B", 12), "sum")
for k in eachindex(Ns)
    println(rpad(Ns[k], 6), rpad(round(mem[k]; sigdigits = 3), 12),
            rpad(round(eAs[k]; sigdigits = 3), 12),
            rpad(round(eBs[k]; sigdigits = 3), 12), round(eSs[k]; sigdigits = 3))
end

p = plot(Ns, max.(mem, 1e-20), yscale = :log10, marker = :circle, lw = 2, label = "memorize (in-train)",
         title = "Error vs series degree N (family of 640 exp ODEs)",
         xlabel = "series degree N", ylabel = "coefficient rel-RMSE")
plot!(p, Ns, max.(eAs, 1e-20), marker = :square, lw = 2, label = "ODE A (out-of-box, λ≈2.85)")
plot!(p, Ns, max.(eBs, 1e-20), marker = :diamond, lw = 2, label = "ODE B (out-of-box)")
plot!(p, Ns, max.(eSs, 1e-20), marker = :utriangle, lw = 2, label = "sum A+B")
savefig(p, "data/n_sweep.png")
println("\nPlot: data/n_sweep.png")
