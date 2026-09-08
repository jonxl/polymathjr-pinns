#=
PAIR TRANSFER (deepest disks).  Train on a PAIR of region disks and test transfer to every
region.  Two training sets, compared:
  * SPIRALS: stable_spiral + unstable_spiral disks  (above the parabola; center sits between them)
  * NODES:   stable_node   + unstable_node   disks  (below the parabola; center is ABOVE it)

Geometric prediction: spirals -> center is INTERPOLATION (center between the two spirals) ->
good; nodes -> center must cross the parabola AND climb in Δ -> EXTRAPOLATION -> worse.  Also
tests stable-vs-unstable transfer within each cross-family target.

Disks placed at the DEEPEST interior point of each region (pole of inaccessibility). Unified-eig
model; metric = solution-space rel-L2.  Robust training (LBFGS may fall back to Adam far out).
Output: data/pair_transfer_deepest.png  (spiral & node maps + transfer bars) + printed table.
=#

using Lux
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote, ComponentArrays
using Plots
import Random
using Random: MersenneTwister
using Statistics: median, mean
using LinearAlgebra

isdir("data") || mkpath("data")
F = Float32
Random.seed!(1234)

# ---- config ----
a0 = F(1.0); a1 = F(0.0)
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)
x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))
xfine = collect(range(x_left, x_right, length = 200)); Mf = length(xfine)
eps_tol = 0.10
Lmap = 4.5; Ng = 91

R_patch = F(0.5); n_patch = 400
loss_floor = F(1e-6); patch_adam = 30000; patch_lbfgs = 60000
centers = Dict(                                        # pole of inaccessibility within [-4.5,4.5]^2
    :saddle          => (F(0.0),  F(-2.25)),
    :stable_node     => (F(-3.45), F(1.05)),
    :unstable_node   => (F(3.45),  F(1.05)),
    :stable_spiral   => (F(-1.55), F(2.9)),
    :unstable_spiral => (F(1.55),  F(2.9)),
    :center          => (F(0.0),  F(2.25)),
)
train_pairs = [(:spirals, [:stable_spiral, :unstable_spiral]),
               (:nodes,   [:stable_node,   :unstable_node])]
eval_order  = [:stable_spiral, :unstable_spiral, :center, :stable_node, :unstable_node, :saddle]

# ---- region + unified-eig helpers ----
function region(t, d)
    d < 0 && return :saddle
    d == 0 && return :degenerate
    disc = t^2 - 4 * d
    t == 0 && return :center
    disc > 0 && return t < 0 ? :stable_node   : :unstable_node
    disc < 0 && return t < 0 ? :stable_spiral : :unstable_spiral
    return :star
end
const Pterm = 14
sxE = [F.(xs    .^ (2n))     ./ F(factorial(big(2n)))     for n in 0:Pterm]
sxO = [F.(xs    .^ (2n + 1)) ./ F(factorial(big(2n + 1))) for n in 0:Pterm]
fxE = [F.(xfine .^ (2n))     ./ F(factorial(big(2n)))     for n in 0:Pterm]
fxO = [F.(xfine .^ (2n + 1)) ./ F(factorial(big(2n + 1))) for n in 0:Pterm]
CS_xs(k) = (sum(sxE[n+1] .* (k .^ n) for n in 0:Pterm), sum(sxO[n+1] .* (k .^ n) for n in 0:Pterm))
Cfun(k, x) = k >= 0 ? cosh(sqrt(k) * x) : cos(sqrt(-k) * x)
Sfun(k, x) = abs(k) < 1e-12 ? x : (k > 0 ? sinh(sqrt(k) * x) / sqrt(k) : sin(sqrt(-k) * x) / sqrt(-k))
function utrue_vals(t, d, xpts)
    μ = Float64(t)/2; k = Float64(t)^2/4 - Float64(d); A = Float64(a0); B = Float64(a1) - μ*Float64(a0)
    return F[ exp(μ*Float64(x)) * (A*Cfun(k, Float64(x)) + B*Sfun(k, Float64(x))) for x in xpts ]
end
const GA = F(π * (3 - sqrt(5)))
function disk_at(cx, cy)
    ts = Vector{F}(undef, n_patch); ds = Vector{F}(undef, n_patch)
    for i in 1:n_patch
        r = R_patch * sqrt((F(i) - F(0.5)) / n_patch); θ = GA * i
        ts[i] = cx + r * cos(θ); ds[i] = cy + r * sin(θ)
    end
    return ts, ds
end
function patch_points(reg)
    cx, cy = centers[reg]
    reg == :center && return fill(F(0.0), n_patch), F.(collect(range(cy - R_patch, cy + R_patch, length = n_patch)))
    return disk_at(cx, cy)
end

net = Lux.Chain(Lux.Dense(2, 64, tanh), Lux.Dense(64, 64, tanh), Lux.Dense(64, 64, tanh), Lux.Dense(64, 4))
_, st = Lux.setup(Random.default_rng(), net)
function batched_u(μ, k, A, B)
    G = length(μ); C = zeros(F, Mf, G); S = zeros(F, Mf, G)
    for n in 0:Pterm
        kn = k .^ n; C .+= fxE[n+1] .* kn; S .+= fxO[n+1] .* kn
    end
    return exp.(xfine * μ) .* (A .* C .+ B .* S)
end
loss_core(p, Xtr, Utr, ntr) = begin
    O = first(net(Xtr, p, st)); μ = O[1:1, :]; k = O[2:2, :]; A = O[3:3, :]; B = O[4:4, :]
    τ = Xtr[1:1, :]; Δ = Xtr[2:2, :]
    C, S = CS_xs(k); v = A .* C .+ B .* S; vp = A .* (k .* S) .+ B .* C
    E = exp.(xs * μ); U = E .* v
    resid = E .* ((μ .^ 2 .+ k .- τ .* μ .+ Δ) .* v .+ (2 .* μ .- τ) .* vp)
    lp = sum(abs2, resid) / (n_colloc * ntr)
    li = (sum(abs2, A .- a0) + sum(abs2, (μ .* A .+ B) .- a1)) / ntr
    ls = sum(abs2, U .- Utr) / (n_colloc * ntr)
    pde_weight * lp + ic_weight * li + sup_weight * ls
end
function train_to_floor(Xtr, Utr, ntr)
    p0, _ = Lux.setup(MersenneTwister(1234), net); p0ca = ComponentArray(p0)
    prob = OptimizationProblem(OptimizationFunction((p, _) -> loss_core(p, Xtr, Utr, ntr), Optimization.AutoZygote()), p0ca)
    stopcb = (s, l) -> l < loss_floor
    r = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); callback = stopcb, maxiters = patch_adam)
    padam = r.u; pfinal = padam
    try
        rl = solve(remake(prob; u0 = padam), OptimizationOptimJL.LBFGS(); callback = stopcb, maxiters = patch_lbfgs)
        isfinite(loss_core(rl.u, Xtr, Utr, ntr)) && (pfinal = rl.u)
    catch
    end
    return pfinal, Float64(loss_core(pfinal, Xtr, Utr, ntr))
end
function pt_err(p, t, d)
    O = vec(first(net(reshape(F[t, d], 2, 1), p, st))); μ, k, A, B = O[1], O[2], O[3], O[4]
    up = [exp(μ*x)*(A*Cfun(k, x) + B*Sfun(k, x)) for x in xfine]; ut = utrue_vals(t, d, xfine)
    sqrt(sum(abs2, up .- ut) / (sum(abs2, ut) + 1e-12))
end
region_error(p, reg) = begin
    ts, ds = patch_points(reg); mean(pt_err(p, ts[k], ds[k]) for k in 1:length(ts))
end

# plane grid + true solution
τg = collect(range(-Lmap, Lmap, length = Ng)); Δg = collect(range(-Lmap, Lmap, length = Ng))
Xgrid = reduce(hcat, [F[t, d] for d in Δg for t in τg]); Gpts = size(Xgrid, 2)
let μt = Xgrid[1:1, :] ./ 2, kt = Xgrid[1:1, :] .^ 2 ./ 4 .- Xgrid[2:2, :]
    global Utrue_grid = batched_u(μt, kt, fill(a0, 1, Gpts), a1 .- μt .* a0)
    global nrm_true = sum(abs2, Utrue_grid; dims = 1) .+ F(1e-12)
end
function plane_error_map(p)
    O = first(net(Xgrid, p, st))
    Upred = batched_u(O[1:1, :], O[2:2, :], O[3:3, :], O[4:4, :])
    errs = vec(sqrt.(sum(abs2, Upred .- Utrue_grid; dims = 1) ./ nrm_true))
    return permutedims(reshape(errs, Ng, Ng))
end
function overlay_regions!(pl)
    plot!(pl, [-Lmap, Lmap], [0.0, 0.0]; color = :white, lw = 1, ls = :dash, label = "")
    plot!(pl, [0.0, 0.0], [0.0, Lmap];   color = :white, lw = 1, ls = :dash, label = "")
    tt = collect(range(-2*sqrt(Lmap), 2*sqrt(Lmap), length = 140)); plot!(pl, tt, (tt .^ 2) ./ 4; color = :white, lw = 1, ls = :dash, label = "")
    return pl
end

# ===========================================================================
# train each pair, evaluate transfer, build map + bars
# ===========================================================================
panels = Plots.Plot[]; results = Dict{Symbol,Dict{Symbol,Float64}}()
for (setname, members) in train_pairs
    ts = F[]; ds = F[]
    for reg in members
        a, b = patch_points(reg); append!(ts, a); append!(ds, b)
    end
    ntr = length(ts); Xtr = permutedims(hcat(ts, ds)); Utr = reduce(hcat, [utrue_vals(ts[k], ds[k], xs) for k in 1:ntr])
    println("training on $setname disks ($(members)) n=$ntr")
    p, fl = train_to_floor(Xtr, Utr, ntr)
    println("  final train loss = ", round(fl; sigdigits = 3), fl < F(1e-5) ? "  (converged)" : "  (UNDERTRAINED)")
    errs = Dict(reg => region_error(p, reg) for reg in eval_order)
    results[setname] = errs
    for reg in eval_order
        tag = reg in members ? "(train)" : (reg == :center ? "(mid)" : "")
        println("    ", rpad(string(reg), 16), "rel-L2 = ", round(errs[reg]; sigdigits = 3), "  ", tag)
    end

    # map
    Emap = plane_error_map(p)
    hm = heatmap(τg, Δg, log10.(max.(Emap, 1e-4)); c = :viridis, clims = (-4, 0.5), colorbar = true, colorbar_title = "log10 rel-L2",
                 xlabel = "τ", ylabel = "Δ", title = "trained on $setname disks", titlefontsize = 9)
    contour!(hm, τg, Δg, Emap; levels = [eps_tol], color = :red, lw = 2, colorbar_entry = false)
    overlay_regions!(hm)
    scatter!(hm, ts, ds; ms = 1.4, color = :white, alpha = 0.5, markerstrokewidth = 0, label = "")
    for reg in eval_order
        reg in members && continue
        cx, cy = centers[reg]; scatter!(hm, [cx], [cy]; ms = 6, color = :orange, markershape = :star5, label = "")
    end
    push!(panels, hm)

    # bars
    cols = [reg in members ? :seagreen : (reg == :center ? :goldenrod : :firebrick) for reg in eval_order]
    xp = 1:length(eval_order)
    bp = bar(xp, [max(errs[reg], 1e-4) for reg in eval_order]; yscale = :log10, color = cols, legend = false,
             xticks = (xp, string.(eval_order)), xrotation = 25, ylabel = "mean rel-L2 (log)",
             title = "$setname transfer (green=train, gold=center)", titlefontsize = 9, bottom_margin = 12Plots.mm)
    hline!(bp, [eps_tol]; color = :red, ls = :dash, label = "")
    push!(panels, bp)
end
savefig(plot(panels...; layout = (2, 2), size = (1500, 1050), left_margin = 5Plots.mm, bottom_margin = 6Plots.mm),
        "data/pair_transfer_deepest.png")

println("\n=== spirals vs nodes: transfer to shared targets ===")
println(rpad("target", 16), rpad("spiral-trained", 16), "node-trained")
for reg in [:center, :saddle]
    println(rpad(string(reg), 16), rpad(round(results[:spirals][reg]; sigdigits = 3), 16), round(results[:nodes][reg]; sigdigits = 3))
end
println("\nPlot: data/pair_transfer_deepest.png")
