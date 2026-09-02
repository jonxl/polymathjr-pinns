#=
SPIRAL -> NODE / CENTER TRANSFER.  Unified-eig model, u = e^{mu x}(A C(k,x)+B S(k,x)),
k = tau^2/4 - Delta  (covers real + complex roots smoothly).

Hypothesis: train on BOTH spirals (stable + unstable) and the net transfers to the nodes and
the center.  Geometry makes a sharp, split prediction:
  * CENTER (tau=0, Delta>0) sits exactly BETWEEN the two spirals -> INTERPOLATION -> predict good.
  * NODES live BELOW the parabola Delta=tau^2/4 (real roots); spirals live ABOVE it (complex
    roots).  Spiral->node crosses the discriminant k=tau^2/4-Delta through 0 -> EXTRAPOLATION.
    The true solution is analytic across k=0 (C,S entire in k), so it MIGHT transfer.

Setup: training set = stable-spiral disk  U  unstable-spiral disk (same fixed R_patch sunflower
disks as family_disks.jl).  Fixed ICs y(0)=1,y'(0)=0.  Train to a loss floor.
Outputs:
  data/spiral_transfer_map.png   full-plane error map + region boundaries + both training disks
  data/spiral_transfer_bars.png  mean rel-L2 on each region (spirals = in-family reference)
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
a0 = F(1.0); a1 = F(0.0)                               # fixed ICs (vary the ODE)
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)
x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))
xfine = collect(range(x_left, x_right, length = 200)); Mf = length(xfine)
eps_tol = 0.10
Lmap = 4.0; Ng = 81

R_patch    = F(0.5); n_patch = 400
loss_floor = F(1e-6)
patch_adam = 30000; patch_lbfgs = 60000
regions = [:saddle, :stable_node, :unstable_node, :stable_spiral, :unstable_spiral, :center]
centers = Dict(
    :saddle          => (F(0.0), F(-1.5)),
    :stable_node     => (F(-3.0), F(1.0)),
    :unstable_node   => (F(3.0),  F(1.0)),
    :stable_spiral   => (F(-1.0), F(2.0)),
    :unstable_spiral => (F(1.0),  F(2.0)),
    :center          => (F(0.0),  F(1.5)),
)

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
    r = solve(remake(prob; u0 = r.u), OptimizationOptimJL.LBFGS(); callback = stopcb, maxiters = patch_lbfgs)
    return r.u, Float64(loss_core(r.u, Xtr, Utr, ntr))
end

# ---- plane grid + true solution ----
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
    return permutedims(reshape(errs, Ng, Ng)), errs
end
function pt_err(p, t, d)
    O = vec(first(net(reshape(F[t, d], 2, 1), p, st))); μ, k, A, B = O[1], O[2], O[3], O[4]
    up = [exp(μ*x)*(A*Cfun(k, x) + B*Sfun(k, x)) for x in xfine]; ut = utrue_vals(t, d, xfine)
    sqrt(sum(abs2, up .- ut) / (sum(abs2, ut) + 1e-12))
end
region_error(p, reg) = begin
    ts, ds = patch_points(reg); e = [pt_err(p, ts[k], ds[k]) for k in 1:length(ts)]
    (mean(e), median(e))
end
function overlay_regions!(pl)
    plot!(pl, [-Lmap, Lmap], [0.0, 0.0]; color = :white, lw = 1, ls = :dash, label = "")
    plot!(pl, [0.0, 0.0], [0.0, Lmap];   color = :white, lw = 1, ls = :dash, label = "")
    tt = collect(range(-2*sqrt(Lmap), 2*sqrt(Lmap), length = 120))
    plot!(pl, tt, (tt .^ 2) ./ 4; color = :white, lw = 1, ls = :dash, label = "")
    return pl
end

# ===========================================================================
# Train on BOTH spiral disks, then evaluate transfer to every region
# ===========================================================================
ts_s, ds_s = patch_points(:stable_spiral); ts_u, ds_u = patch_points(:unstable_spiral)
ts = vcat(ts_s, ts_u); ds = vcat(ds_s, ds_u); ntr = length(ts)
Xtr = permutedims(hcat(ts, ds)); Utr = reduce(hcat, [utrue_vals(ts[k], ds[k], xs) for k in 1:ntr])
println("training on stable+unstable spiral disks (n=$ntr), train-to-floor<$loss_floor")
p, fl = train_to_floor(Xtr, Utr, ntr)
println("final train loss = ", round(fl; sigdigits = 3), fl < F(1e-5) ? "  (converged)" : "  (UNDERTRAINED)")

# per-region transfer error (spirals = in-family reference)
order = [:stable_spiral, :unstable_spiral, :center, :stable_node, :unstable_node, :saddle]
tags  = Dict(:stable_spiral => "stable spiral\n(train)", :unstable_spiral => "unstable spiral\n(train)",
             :center => "center\n(interp)", :stable_node => "stable node\n(extrap)",
             :unstable_node => "unstable node\n(extrap)", :saddle => "saddle\n(far)")
println("\nregion            mean rel-L2   median rel-L2   (spirals = in-family)")
means = Float64[]
for reg in order
    m, md = region_error(p, reg); push!(means, m)
    println(rpad(string(reg), 16), rpad(round(m; sigdigits = 3), 14), round(md; sigdigits = 3))
end

# [1] full-plane error map
Emap, _ = plane_error_map(p)
hm = heatmap(τg, Δg, log10.(max.(Emap, 1e-4)); c = :viridis, colorbar_title = "log10 rel-L2",
             xlabel = "τ", ylabel = "Δ", title = "trained on BOTH spirals: error over the plane, ε=$eps_tol")
contour!(hm, τg, Δg, Emap; levels = [eps_tol], color = :red, lw = 2, colorbar_entry = false)
overlay_regions!(hm)
scatter!(hm, ts, ds; ms = 1.5, color = :white, alpha = 0.5, markerstrokewidth = 0, label = "train disks")
# mark the evaluation centers of the transfer targets
for reg in [:center, :stable_node, :unstable_node, :saddle]
    cx, cy = centers[reg]; scatter!(hm, [cx], [cy]; ms = 5, color = :orange, markershape = :star5, label = "")
end
savefig(hm, "data/spiral_transfer_map.png")

# [2] transfer bar chart
xp = collect(1:length(order))
cols = [reg in (:stable_spiral, :unstable_spiral) ? :seagreen : (reg == :center ? :goldenrod : :firebrick) for reg in order]
bp = bar(xp, max.(means, 1e-4); yscale = :log10, color = cols, legend = false,
         xticks = (xp, [tags[reg] for reg in order]), xrotation = 20, ylabel = "mean rel-L2 (log)",
         title = "spiral-trained net: transfer error by region", bottom_margin = 14Plots.mm)
hline!(bp, [eps_tol]; color = :red, ls = :dash, label = "")
savefig(bp, "data/spiral_transfer_bars.png")
println("\nPlots: data/spiral_transfer_map.png , data/spiral_transfer_bars.png")
