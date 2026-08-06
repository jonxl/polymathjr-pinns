#=
Sweep: train a separate model on EACH trace-determinant section and report the
per-section coefficient error. Produces a summary table and a comparison bar chart.

Same machinery as generalize_pinn.jl (network (tau,Delta) -> c_n, three weighted loss
terms), just looped over the 6 regions with a fresh model each time.

Default weights below are PURE SUPERVISED (pde=0, ic=0, sup=1) -- the clean,
presentable numbers. Flip to pde/ic for the PINN version.
=#

using Lux
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote
using ComponentArrays
using Plots, ProgressMeter
import Random
using Random: MersenneTwister
using LinearAlgebra

isdir("data") || mkpath("data")
F = Float32

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N  = 10
a0 = F(1.0); a1 = F(0.0)

# loss-term knobs (default = pure supervised)
pde_weight = F(1.0)
ic_weight  = F(1.0)
sup_weight = F(1.0)

tau_lim   = F(2.0)
delta_lim = F(2.0)
n_total   = 600

adam_iters  = 100000
lbfgs_iters = 100000

x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))
fact_vec = F.(factorial.(0:N))

# the six sections to sweep
regions = [:saddle, :stable_node, :unstable_node, :stable_spiral, :unstable_spiral, :center]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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

# Sample n ODEs from one region (center is a measure-zero line: special-cased to tau=0).
function sample_region(reg, n, rng)
    ts = F[]; ds = F[]
    if reg == :center
        while length(ts) < n
            push!(ts, F(0.0)); push!(ds, rand(rng, F) * delta_lim)   # tau=0, Delta>0
        end
    else
        while length(ts) < n
            t = rand(rng, F) * (2 * tau_lim)   - tau_lim
            d = rand(rng, F) * (2 * delta_lim) - delta_lim
            region(t, d) == reg && (push!(ts, t); push!(ds, d))
        end
    end
    return ts, ds
end

# Power matrices (shared collocation grid).
Pu  = F[ xs[m]^(i - 1)                                    for m in 1:n_colloc, i in 1:N+1 ]
Pu1 = F[ (i - 1) >= 1 ? (i - 1) * xs[m]^(i - 2) : 0       for m in 1:n_colloc, i in 1:N+1 ]
Pu2 = F[ (i - 1) >= 2 ? (i - 1)*(i - 2)*xs[m]^(i - 3) : 0 for m in 1:n_colloc, i in 1:N+1 ]

# Network architecture (re-initialized fresh for each region).
net = Lux.Chain(Lux.Dense(2, 64, tanh), Lux.Dense(64, 64, tanh),
                Lux.Dense(64, 64, tanh), Lux.Dense(64, N + 1))
_, st = Lux.setup(Random.default_rng(), net)   # st is stateless/constant

function pinn_loss(p, Xb, Yb, nb)
    C = first(net(Xb, p, st))
    loss = sup_weight == 0 ? zero(F) : sup_weight * sum(abs2, C .- Yb) / ((N + 1) * nb)
    if pde_weight != 0
        τ = Xb[1:1, :]; Δ = Xb[2:2, :]
        resid = (Pu2 * C) .- τ .* (Pu1 * C) .+ Δ .* (Pu * C)
        loss += pde_weight * sum(abs2, resid) / (n_colloc * nb)
    end
    if ic_weight != 0
        loss += ic_weight * (sum(abs2, C[1, :] .- a0) + sum(abs2, C[2, :] .- a1)) / nb
    end
    return loss
end

rel_rmse(p, Xb, Yb, nb) = sqrt(sum(sum(abs2, first(net(Xb, p, st)) .- Yb; dims = 1) ./
                                   (sum(abs2, Yb; dims = 1) .+ F(1e-8))) / nb)

# ---------------------------------------------------------------------------
# Train + evaluate one region
# ---------------------------------------------------------------------------
function run_region(reg; seed = 1234)
    rng = MersenneTwister(seed)
    ts, ds = sample_region(reg, n_total, rng)
    Xr = permutedims(hcat(ts, ds))
    Yr = reduce(hcat, [true_cn(ts[k], ds[k]) for k in 1:n_total])

    perm = Random.shuffle(rng, 1:n_total)
    nte  = round(Int, 0.2 * n_total)
    te, tr = perm[1:nte], perm[nte+1:end]
    Xtr, Ytr = Xr[:, tr], Yr[:, tr]
    Xte, Yte = Xr[:, te], Yr[:, te]
    ntr, nteN = length(tr), length(te)

    p0, _ = Lux.setup(rng, net)
    p0ca  = ComponentArray(p0)
    lf(p, _) = pinn_loss(p, Xtr, Ytr, ntr)

    th = Float64[]; vh = Float64[]                 # train / test rel-rmse history
    cb = function (state, l)
        push!(th, rel_rmse(state.u, Xtr, Ytr, ntr))
        push!(vh, rel_rmse(state.u, Xte, Yte, nteN))
        return false
    end
    prob  = OptimizationProblem(OptimizationFunction(lf, Optimization.AutoZygote()), p0ca)
    r     = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); callback = cb, maxiters = adam_iters)
    nadam = length(th)
    r     = solve(remake(prob; u0 = r.u), OptimizationOptimJL.LBFGS(); callback = cb, maxiters = lbfgs_iters)

    pred = first(net(Xr, r.u, st))                 # per-point error over the whole region
    rerr = vec(sqrt.(sum(abs2, pred .- Yr; dims = 1) ./ (sum(abs2, Yr; dims = 1) .+ F(1e-8))))
    istest = falses(n_total); istest[te] .= true

    return (train_hist = th, test_hist = vh, n_adam = nadam,
            n_lbfgs = length(th) - nadam, retcode = r.retcode,
            taus = ts, deltas = ds, rel_err = rerr, is_test = istest,
            train = th[end], test = vh[end])
end

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
println("weights: pde=$pde_weight ic=$ic_weight sup=$sup_weight\n")
results = []
for reg in regions
    print("training $reg ... ")
    res = run_region(reg)
    push!(results, res)
    println("train=", round(res.train; sigdigits = 4), "  test=", round(res.test; sigdigits = 4),
            "   [LBFGS ran ", res.n_lbfgs, "/", lbfgs_iters, " iters, ", res.retcode, "]")
end

println("\n", rpad("region", 18), rpad("train rel-rmse", 18), "test rel-rmse")
for (i, reg) in enumerate(regions)
    println(rpad(string(reg), 18), rpad(round(results[i].train; sigdigits = 4), 18),
            round(results[i].test; sigdigits = 4))
end

# ---------------------------------------------------------------------------
# Figure 1: 6 loss plots (train & test rel-rmse vs iteration), one per section
# ---------------------------------------------------------------------------
loss_panels = Plots.Plot[]
for (i, reg) in enumerate(regions)
    rr = results[i]
    p = plot(1:length(rr.train_hist), max.(rr.train_hist, 1e-20), yscale = :log10,
             lw = 2, label = (i == 1 ? "train" : ""), title = string(reg),
             xlabel = "iteration", ylabel = "rel-rmse", legend = (i == 1 ? :topright : false))
    plot!(p, 1:length(rr.test_hist), max.(rr.test_hist, 1e-20), lw = 2,
          label = (i == 1 ? "test" : ""))
    vline!(p, [rr.n_adam + 0.5], ls = :dash, color = :gray, label = "")
    push!(loss_panels, p)
end
loss_fig = plot(loss_panels...; layout = (2, 3), size = (1300, 720))
savefig(loss_fig, "data/sweep_loss_grid.png")

# ---------------------------------------------------------------------------
# Figure 2: 6 error maps (per-point coeff error over the τ-Δ plane), one per section
# ---------------------------------------------------------------------------
τg = range(-tau_lim, tau_lim, length = 200)
map_panels = Plots.Plot[]
for (i, reg) in enumerate(regions)
    rr = results[i]
    le = log10.(max.(rr.rel_err, F(1e-12)))
    p = scatter(rr.taus[.!rr.is_test], rr.deltas[.!rr.is_test]; marker_z = le[.!rr.is_test],
                markershape = :circle, ms = 3, label = (i == 1 ? "train" : ""),
                title = string(reg), xlabel = "τ", ylabel = "Δ",
                colorbar_title = "log10 rel err", legend = (i == 1 ? :topright : false),
                xlims = (-tau_lim, tau_lim), ylims = (-delta_lim, delta_lim))
    scatter!(p, rr.taus[rr.is_test], rr.deltas[rr.is_test]; marker_z = le[rr.is_test],
             markershape = :diamond, ms = 5, markerstrokewidth = 1,
             label = (i == 1 ? "test" : ""))
    plot!(p, τg, (τg .^ 2) ./ 4, color = :black, lw = 1.5, label = "")
    hline!(p, [0.0], color = :black, ls = :dash, lw = 1, label = "")
    push!(map_panels, p)
end
map_fig = plot(map_panels...; layout = (2, 3), size = (1500, 850))
savefig(map_fig, "data/sweep_maps_grid.png")

println("\nPlots saved: data/sweep_loss_grid.png , data/sweep_maps_grid.png")
