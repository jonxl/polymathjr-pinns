#=
Cross-region generalization (out-of-family WITHIN 2nd-order linear).

For each trace-determinant section, train a model on ODEs from THAT section only
(in-range box), then test it on every section. Produces a 6x6 transfer matrix:
    entry (i, j) = mean coeff rel-error when trained on region i, tested on region j.
  - diagonal  = in-family (same section)            -> low
  - off-diag  = out-of-family (different section)    -> high; row i = "train on i, test elsewhere"

Same (tau,Delta) input as before (no encoding change -- all sections are 2nd-order
linear), same PINN config as generalize_pinn.jl.
=#

using Lux
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote, ComponentArrays
using Plots, ProgressMeter
import Random
using Random: MersenneTwister
using Statistics: mean
using LinearAlgebra

isdir("data") || mkpath("data")
F = Float32

# ---------------------------------------------------------------------------
# Config (same as generalize_pinn.jl)
# ---------------------------------------------------------------------------
N  = 10
a0 = F(1.0); a1 = F(0.0)
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)

tau_lim = F(2.0); delta_lim = F(2.0)     # in-range box (all sections sampled here)
n_train = 600
n_test  = 300
x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))

adam_iters  = 100000
lbfgs_iters = 100000

regions = [:saddle, :stable_node, :unstable_node, :stable_spiral, :unstable_spiral, :center]

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

# sample n ODEs from a section, inside the in-range box (center = tau=0 line)
function sample_region(reg, n, rng)
    ts = F[]; ds = F[]
    if reg == :center
        while length(ts) < n
            push!(ts, F(0.0)); push!(ds, rand(rng, F) * delta_lim)
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

data_of(ts, ds) = (permutedims(hcat(ts, ds)),
                   reduce(hcat, [true_cn(ts[k], ds[k]) for k in 1:length(ts)]))

Pu  = F[ xs[m]^(i - 1)                                    for m in 1:n_colloc, i in 1:N+1 ]
Pu1 = F[ (i - 1) >= 1 ? (i - 1) * xs[m]^(i - 2) : 0       for m in 1:n_colloc, i in 1:N+1 ]
Pu2 = F[ (i - 1) >= 2 ? (i - 1)*(i - 2)*xs[m]^(i - 3) : 0 for m in 1:n_colloc, i in 1:N+1 ]

net = Lux.Chain(Lux.Dense(2, 64, tanh), Lux.Dense(64, 64, tanh),
                Lux.Dense(64, 64, tanh), Lux.Dense(64, N + 1))
_, st = Lux.setup(Random.default_rng(), net)

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

# the three loss terms returned SEPARATELY (raw / unweighted), for any (Xb, Yb)
function loss_components(p, Xb, Yb, nb)
    C  = first(net(Xb, p, st))
    τ  = Xb[1:1, :]; Δ = Xb[2:2, :]
    lp = sum(abs2, (Pu2 * C) .- τ .* (Pu1 * C) .+ Δ .* (Pu * C)) / (n_colloc * nb)
    li = (sum(abs2, C[1, :] .- a0) + sum(abs2, C[2, :] .- a1)) / nb
    ls = sum(abs2, C .- Yb) / ((N + 1) * nb)
    return lp, li, ls
end

# ---------------------------------------------------------------------------
# Train one model per region
# ---------------------------------------------------------------------------
trained = Dict{Symbol,Any}()
for reg in regions
    rng = MersenneTwister(1234)
    ts, ds = sample_region(reg, n_train, rng)
    Xtr, Ytr = data_of(ts, ds)
    p0, _ = Lux.setup(rng, net); p0ca = ComponentArray(p0)
    lf(p, _) = pinn_loss(p, Xtr, Ytr, n_train)
    prob = OptimizationProblem(OptimizationFunction(lf, Optimization.AutoZygote()), p0ca)
    pb = Progress(adam_iters, desc = "train $reg Adam ")
    cb = (s, l) -> (next!(pb; showvalues = [(:loss, l)]); false)
    r = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); callback = cb, maxiters = adam_iters)
    r = solve(remake(prob; u0 = r.u), OptimizationOptimJL.LBFGS(); maxiters = lbfgs_iters)
    trained[reg] = r.u
    println("  trained on $reg")
end

# ---------------------------------------------------------------------------
# Build the 6x6 transfer matrix: train region (row) x test region (col)
# ---------------------------------------------------------------------------
rng = MersenneTwister(99)
test_sets = Dict(reg => data_of(sample_region(reg, n_test, rng)...) for reg in regions)

M    = zeros(length(regions), length(regions))   # coeff rel-rmse (as before)
Mpde = zeros(length(regions), length(regions))   # PDE residual component
Mic  = zeros(length(regions), length(regions))   # IC component
Msup = zeros(length(regions), length(regions))   # supervised component
for (i, tr_reg) in enumerate(regions), (j, te_reg) in enumerate(regions)
    Xte, Yte = test_sets[te_reg]
    M[i, j] = rel_rmse(trained[tr_reg], Xte, Yte, n_test)
    lp, li, ls = loss_components(trained[tr_reg], Xte, Yte, n_test)
    Mpde[i, j] = lp; Mic[i, j] = li; Msup[i, j] = ls
end

# ---------------------------------------------------------------------------
# Report a labeled table + log10 heatmap for any train×test matrix.
# ---------------------------------------------------------------------------
labels = string.(regions)
function report_matrix(Mat, fname, ttl; pct = false)
    println("\nrows = trained on, cols = tested on  ($ttl)")
    println(rpad("", 18), join([rpad(string(r), 16) for r in regions]))
    for (i, r) in enumerate(regions)
        println(rpad(string(r), 18),
                join([rpad(round(Mat[i, j]; sigdigits = 3), 16) for j in 1:length(regions)]))
    end
    hm = heatmap(labels, labels, log10.(max.(Mat, 1e-20));
                 yflip = true, c = :viridis, colorbar_title = "log10 value",
                 xlabel = "tested on", ylabel = "trained on", xrotation = 30,
                 title = ttl, size = (820, 680))
    for i in 1:length(regions), j in 1:length(regions)
        txt = pct ? string(round(Mat[i, j] * 100; sigdigits = 2), "%") :
                    string(round(Mat[i, j]; sigdigits = 2))
        annotate!(hm, j, i, text(txt, 6, :white))
    end
    savefig(hm, fname)
    println("Plot saved: $fname")
end

report_matrix(M,    "data/cross_region_matrix.png", "rel-rmse (diag = in-family)"; pct = true)
report_matrix(Mpde, "data/cross_region_pde.png",    "PDE residual component (raw)")
report_matrix(Mic,  "data/cross_region_ic.png",     "IC component (raw)")
report_matrix(Msup, "data/cross_region_sup.png",    "supervised component (raw)")

# ---------------------------------------------------------------------------
# Grouped-bar summary (non-heatmap): in-family (diagonal) vs out-of-family
# (off-diagonal), GEOMETRIC mean per component -- since values span many orders
# of magnitude, geomean gives the typical cell, not one dominated by the max.
# The point: out-of-family, PDE residual blows up far more than IC or supervised.
# ---------------------------------------------------------------------------
geomean(v)        = exp(mean(log.(max.(v, 1e-20))))
diag_vals(Mat)    = [Mat[i, i] for i in 1:nr]
offdiag_vals(Mat) = [Mat[i, j] for i in 1:nr for j in 1:nr if i != j]
comp_mats  = [Mpde, Mic, Msup]; comp_names = ["PDE residual", "IC", "supervised"]
in_fam  = [geomean(diag_vals(Mat))    for Mat in comp_mats]
out_fam = [geomean(offdiag_vals(Mat)) for Mat in comp_mats]
println("\ncomponent      in-family      out-of-family    blow-up")
for k in 1:3
    println(rpad(comp_names[k], 15), rpad(round(in_fam[k]; sigdigits = 3), 15),
            rpad(round(out_fam[k]; sigdigits = 3), 17), "×", round(out_fam[k] / in_fam[k]; sigdigits = 2))
end
xp = collect(1:3)
bpc = bar(xp .- 0.2, in_fam; bar_width = 0.4, label = "in-family (diagonal)", color = :seagreen,
          yscale = :log10, xticks = (xp, comp_names), legend = :topright,
          ylabel = "loss component (raw, geometric mean)",
          title = "Cross-region loss components: in-family vs out-of-family", size = (860, 560))
bar!(bpc, xp .+ 0.2, out_fam; bar_width = 0.4, label = "out-of-family (off-diagonal)", color = :firebrick)
for k in 1:3
    annotate!(bpc, k, out_fam[k] * 4, text("×$(round(out_fam[k]/in_fam[k]; sigdigits = 2))", 8))
end
savefig(bpc, "data/cross_region_component_bars.png")
println("Plot saved: data/cross_region_component_bars.png")
