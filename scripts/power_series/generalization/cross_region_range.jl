#=
Out-of-range AND out-of-family combined -- swept over ALL training sections.

For EACH section, train a model on that section in-range (max(|tau|,|Delta|)<=train_lim),
then test on every section across shells R=1..Rmax. One panel per training section
(6 panels in a 2x3 grid). In each panel:
  x = R,  y = mean rel-error,  one line per test section.
    - bold solid line = the TRAINING section (in-family)
    - dashed lines     = other sections (out-of-family)
So vertical gaps = out-of-family penalty; rise along a line = out-of-range penalty.

Same (tau,Delta) input and PINN config as generalize_pinn.jl.
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
# Config
# ---------------------------------------------------------------------------
N  = 10
a0 = F(1.0); a1 = F(0.0)
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)

train_lim = F(2.0)
n_train   = 600
x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))

adam_iters  = 2000
lbfgs_iters = 5000
Rmax        = 8
n_per_shell = 200

regions = [:saddle, :stable_node, :unstable_node, :stable_spiral, :unstable_spiral, :center]
Rs = collect(1:Rmax)

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

function sample_box(reg, n, rng)
    ts = F[]; ds = F[]
    if reg == :center
        while length(ts) < n
            push!(ts, F(0.0)); push!(ds, rand(rng, F) * train_lim)
        end
    else
        while length(ts) < n
            t = rand(rng, F) * (2 * train_lim) - train_lim
            d = rand(rng, F) * (2 * train_lim) - train_lim
            region(t, d) == reg && (push!(ts, t); push!(ds, d))
        end
    end
    return ts, ds
end

function sample_shell(reg, R, n, rng)
    ts = F[]; ds = F[]
    if reg == :center
        while length(ts) < n
            push!(ts, F(0.0)); push!(ds, (R - 1) + rand(rng, F))
        end
    else
        while length(ts) < n
            t = rand(rng, F) * (2R) - R
            d = rand(rng, F) * (2R) - R
            m = max(abs(t), abs(d))
            ((R - 1) < m <= R) && region(t, d) == reg && (push!(ts, t); push!(ds, d))
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

# ---------------------------------------------------------------------------
# Train on one section (in-range), then test every section x every shell
# ---------------------------------------------------------------------------
function run_train_region(train_reg)
    rng = MersenneTwister(1234)
    ts, ds = sample_box(train_reg, n_train, rng)
    Xtr, Ytr = data_of(ts, ds)
    p0, _ = Lux.setup(rng, net); p0ca = ComponentArray(p0)
    lf(p, _) = pinn_loss(p, Xtr, Ytr, n_train)
    prob = OptimizationProblem(OptimizationFunction(lf, Optimization.AutoZygote()), p0ca)
    pb = Progress(adam_iters, desc = "train $train_reg Adam ")
    cb = (s, l) -> (next!(pb; showvalues = [(:loss, l)]); false)
    r = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); callback = cb, maxiters = adam_iters)
    r = solve(remake(prob; u0 = r.u), OptimizationOptimJL.LBFGS(); maxiters = lbfgs_iters)
    pt = r.u

    curves = Dict{Symbol,Vector{Float64}}()
    for reg in regions
        errs = Float64[]
        for R in Rs
            tts, tds = sample_shell(reg, F(R), n_per_shell, rng)
            Xs, Ys = data_of(tts, tds)
            push!(errs, rel_rmse(pt, Xs, Ys, n_per_shell))
        end
        curves[reg] = errs
    end
    return curves
end

# ---------------------------------------------------------------------------
# Sweep all training sections -> 2x3 grid
# ---------------------------------------------------------------------------
panels = Plots.Plot[]
for (idx, treg) in enumerate(regions)
    println("=== train on $treg ($idx/$(length(regions))) ===")
    curves = run_train_region(treg)
    p = plot(yscale = :log10, title = "train: $treg", xlabel = "R = max(|τ|,|Δ|)",
             ylabel = "mean rel err", titlefontsize = 10, legendfontsize = 6,
             legend = (idx == 1 ? :topright : false))
    for (k, reg) in enumerate(regions)
        is_tr = reg == treg
        plot!(p, Rs, curves[reg]; color = k, marker = :circle, ms = 2,
              lw = is_tr ? 3 : 1.2, ls = is_tr ? :solid : :dash, label = string(reg))
    end
    vline!(p, [train_lim], color = :black, ls = :dot, label = "")
    push!(panels, p)
end
fig = plot(panels...; layout = (2, 3), size = (1500, 850))
savefig(fig, "data/cross_region_range_grid.png")
println("\nPlot saved: data/cross_region_range_grid.png")
