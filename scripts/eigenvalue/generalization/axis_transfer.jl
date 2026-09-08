#=
AXIS TRANSFER.  Train on a row of disks along one axis of the trace-det plane and test on a
row of disks along the ORTHOGONAL axis.  Two directions, compared:
  * train X (centers on Δ=0, the τ-axis: near-constant/gentle solutions) -> test Y (τ=0, the
    Δ-axis: oscillatory center above, hyperbolic saddle below)
  * train Y -> test X  (reverse)

Geometric prediction: the two axes cross only at the origin, so transfer should hold near the
origin and fail moving out along the test axis (where solutions become qualitatively different:
oscillation up the Δ-axis, hyperbolic growth down it).

Fixed R_patch sunflower disks, unified-eig model, solution-space rel-L2, robust Adam→LBFGS.
Output: data/axis_transfer.png  (train-X and train-Y maps + transfer-error bars) + printed table.
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

a0 = F(1.0); a1 = F(0.0)
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)
x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))
xfine = collect(range(x_left, x_right, length = 200)); Mf = length(xfine)
eps_tol = 0.10
Lmap = 6.0; Ng = 121

R_patch = F(0.5); n_patch = 300
loss_floor = F(1e-6); patch_adam = 30000; patch_lbfgs = 60000
axis_pos = Float64[-3.0, -1.5, 0.0, 1.5, 3.0]         # disk-center positions along an axis
xcenters = [(t, 0.0) for t in axis_pos]               # τ-axis (Δ=0)
ycenters = [(0.0, d) for d in axis_pos]               # Δ-axis (τ=0)

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
        ts[i] = F(cx) + r * cos(θ); ds[i] = F(cy) + r * sin(θ)
    end
    return ts, ds
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
function disk_err(p, cx, cy)
    ts, ds = disk_at(cx, cy)
    return mean(pt_err(p, ts[k], ds[k]) for k in 1:n_patch)
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
    plot!(pl, [0.0, 0.0], [-Lmap, Lmap];   color = :white, lw = 1, ls = :dash, label = "")
    tt = collect(range(-2*sqrt(Lmap), 2*sqrt(Lmap), length = 160)); plot!(pl, tt, (tt .^ 2) ./ 4; color = :white, lw = 1, ls = :dash, label = "")
    return pl
end

# ===========================================================================
# run both directions
# ===========================================================================
runs = [("train X (τ-axis) → test Y (Δ-axis)", xcenters, ycenters, "test Δ (position on Y-axis)"),
        ("train Y (Δ-axis) → test X (τ-axis)", ycenters, xcenters, "test τ (position on X-axis)")]
panels = Plots.Plot[]
for (name, traincs, testcs, xlab) in runs
    ts = F[]; ds = F[]
    for (cx, cy) in traincs
        a, b = disk_at(cx, cy); append!(ts, a); append!(ds, b)
    end
    ntr = length(ts); Xtr = permutedims(hcat(ts, ds)); Utr = reduce(hcat, [utrue_vals(ts[k], ds[k], xs) for k in 1:ntr])
    println("$name  (n=$ntr)")
    p, fl = train_to_floor(Xtr, Utr, ntr)
    println("  final train loss = ", round(fl; sigdigits = 3), fl < F(1e-5) ? "  (converged)" : "  (UNDERTRAINED)")
    errs = [disk_err(p, cx, cy) for (cx, cy) in testcs]
    for ((cx, cy), e) in zip(testcs, errs)
        println("    test disk ($cx,$cy)  [$(region(F(cx), F(cy)))]  rel-L2 = ", round(e; sigdigits = 3))
    end

    # map
    Emap = plane_error_map(p)
    hm = heatmap(τg, Δg, log10.(max.(Emap, 1e-4)); c = :viridis, clims = (-4, 0.5), colorbar = true, colorbar_title = "log10 rel-L2",
                 xlabel = "τ", ylabel = "Δ", title = name, titlefontsize = 9, aspect_ratio = :equal, xlims = (-Lmap, Lmap), ylims = (-Lmap, Lmap))
    contour!(hm, τg, Δg, Emap; levels = [eps_tol], color = :red, lw = 2, colorbar_entry = false)
    overlay_regions!(hm)
    scatter!(hm, ts, ds; ms = 1.1, color = :white, alpha = 0.4, markerstrokewidth = 0, label = "")
    scatter!(hm, [F(c[1]) for c in testcs], [F(c[2]) for c in testcs]; ms = 6, color = :orange, markershape = :star5, label = "")
    push!(panels, hm)

    # bars
    pos = [name[7] == 'X' ? c[2] : c[1] for c in testcs]   # test-axis coordinate
    bp = bar(1:length(testcs), max.(errs, 1e-4); yscale = :log10, color = :steelblue, legend = false,
             xticks = (1:length(testcs), string.(pos)), xlabel = xlab, ylabel = "mean rel-L2 (log)",
             title = "transfer error along test axis", titlefontsize = 9, bottom_margin = 8Plots.mm)
    hline!(bp, [eps_tol]; color = :red, ls = :dash, label = "")
    push!(panels, bp)
end
savefig(plot(panels...; layout = (2, 2), size = (1500, 1050), left_margin = 5Plots.mm, bottom_margin = 6Plots.mm),
        "data/axis_transfer.png")
println("\nPlot: data/axis_transfer.png")
