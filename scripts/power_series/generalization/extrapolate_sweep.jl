#=
Extrapolation sweep: for EACH trace-determinant section, train a PINN on the in-range
box (that section) and test on shells of increasing parameter magnitude R within the
same section. Produces ONE image with 6 panels (error vs R, one per section).

Same PINN config as generalize_pinn.jl / extrapolate_range.jl.
Shell R holds ODEs with R-1 < max(|tau|,|Delta|) <= R; R<=train_lim is in-range.
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

train_lim = F(2.0)
n_train   = 600
x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))

adam_iters  = 100000          # plateaus by here; LBFGS does the precision work
lbfgs_iters = 100000          # self-stops at convergence
Rmax        = 10
n_per_shell = 300

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

# in-range box sample for a region (center = the tau=0 line)
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

# shell of magnitude R for a region
function sample_shell(reg, R, n, rng)
    ts = F[]; ds = F[]
    if reg == :center
        while length(ts) < n
            push!(ts, F(0.0)); push!(ds, (R - 1) + rand(rng, F))   # Delta in (R-1, R]
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

# ---------------------------------------------------------------------------
# Train on one section, test on its shells
# ---------------------------------------------------------------------------
function run_region(reg; seed = 1234)
    rng = MersenneTwister(seed)
    tt, td = sample_box(reg, n_train, rng)
    Xtr = permutedims(hcat(tt, td))
    Ytr = reduce(hcat, [true_cn(tt[k], td[k]) for k in 1:n_train])

    p0, _ = Lux.setup(rng, net); p0ca = ComponentArray(p0)
    lf(p, _) = pinn_loss(p, Xtr, Ytr, n_train)
    prob = OptimizationProblem(OptimizationFunction(lf, Optimization.AutoZygote()), p0ca)

    pb = Progress(adam_iters, desc = "  $reg Adam ")
    cb = (s, l) -> (next!(pb; showvalues = [(:loss, l)]); false)
    r = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); callback = cb, maxiters = adam_iters)
    r = solve(remake(prob; u0 = r.u), OptimizationOptimJL.LBFGS(); maxiters = lbfgs_iters)
    pt = r.u

    errs = Float64[]
    for R in Rs
        ts, ds = sample_shell(reg, F(R), n_per_shell, rng)
        Xs = permutedims(hcat(ts, ds))
        Ys = reduce(hcat, [true_cn(ts[k], ds[k]) for k in 1:n_per_shell])
        C  = first(net(Xs, pt, st))
        re = vec(sqrt.(sum(abs2, C .- Ys; dims = 1) ./ (sum(abs2, Ys; dims = 1) .+ F(1e-8))))
        push!(errs, mean(re))
    end
    return errs
end

# ---------------------------------------------------------------------------
# Sweep + plot
# ---------------------------------------------------------------------------
println("weights: pde=$pde_weight ic=$ic_weight sup=$sup_weight\n")
panels = Plots.Plot[]
for (i, reg) in enumerate(regions)
    println("=== $reg ($i/$(length(regions))) ===")
    errs = run_region(reg)
    p = plot(Rs, errs; marker = :circle, ms = 4, lw = 2, yscale = :log10,
             title = string(reg), xlabel = "R = max(|τ|,|Δ|)", ylabel = "mean rel err",
             label = "", legend = false)
    vline!(p, [train_lim], ls = :dash, color = :red, label = "")
    push!(panels, p)
end
fig = plot(panels...; layout = (2, 3), size = (1300, 720))
savefig(fig, "data/extrapolate_sweep_grid.png")
println("\nPlot saved: data/extrapolate_sweep_grid.png")
