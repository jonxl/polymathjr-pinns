#=
FAMILY GENERALIZATION on a FAIR footing (fixed-disk patches).  Unified-eig model,
u = e^{mu x}(A C(k,x) + B S(k,x)), k = tau^2/4 - Delta  (covers all regions, real + complex).

Why this script exists
----------------------
The old per-family test (gen_radius.jl, run_family) trained each region's net on the WHOLE
region.  But the regions have very different area AND shape (saddle = half-plane, center = a
line, nodes/spirals = curved wedges), so the "generalization area" it reported was dominated
by region geometry, not by how well the family generalizes.  Rejection-sampling the region
also clusters points wherever the region is fat -> spatial bias.

Fix (all regions on identical footing):
  * FIXED-SHAPE patch: every region is trained on a disk of the SAME radius R_patch, centered
    at a canonical interior point, so training coverage is identical across regions.
  * EVEN placement: a sunflower / Fibonacci lattice (r_i = R sqrt((i-.5)/n), th_i = i*GA)
    gives equal-area, deterministic, cluster-free points -- no density bias.
  * CENTER is degenerate (tau=0, Delta>0 is a 1-D ray): it cannot hold a 2-D disk, so it gets
    a 1-D segment patch of the same point count, clearly labelled.
  * TRAIN TO A FLOOR: Adam->LBFGS until training loss < loss_floor (high iter cap as backstop),
    so a large out-of-region error can never be blamed on undertraining.
  * Metric: solution-space relative L2 everywhere.

The fair, geometry-free score is the MEDIAN REACH past the (identical) patch edge -- how far
r(theta) from the patch center gets before rel-L2 crosses eps, minus R_patch.

Two experiments (toggle at top):
  [A] run_maps   : error map over the (tau,Delta) plane + eps-contour + patch, per region.
                   Reports footprint area AND the fair median reach.   -> data/family_disks_maps.png
  [B] run_extrap : from each region's patch center, march radial shells outward; bin ring
                   points by region and plot mean rel-L2 vs distance (in-family vs out-of-family
                   vs out-of-range on one axis).                       -> data/family_disks_extrap.png
=#

using Lux
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote, ComponentArrays
using Plots
import Random
using Random: MersenneTwister
using Statistics: median, mean, std
using LinearAlgebra

isdir("data") || mkpath("data")
F = Float32
Random.seed!(1234)

# ---- what to run ----
run_maps      = true
run_extrap    = true
run_placement = false      # already validated; skip to save compute and avoid overwriting

# ---- shared config ----
a0 = F(1.0); a1 = F(0.0)                               # fixed ICs y(0)=1, y'(0)=0 (vary the ODE)
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)
x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))
xfine = collect(range(x_left, x_right, length = 200)); Mf = length(xfine)
eps_tol = 0.10                                        # reach/contour tolerance (set 0.01 for the strict-tolerance view)
Lmap = 8.0; Ng = 161                                  # error-map half-width + resolution

# ---- fixed-disk patch config ----
R_patch    = F(0.3)                                   # SAME training radius for every region (tighter disk)
n_patch    = 400                                      # SAME point budget for every region
loss_floor = F(1e-7)                                  # early-stop TARGET (Float32 LBFGS floors near here)
flag_thresh = F(1e-5)                                 # warn "undertrained" only if final loss exceeds this
patch_adam = 30000; patch_lbfgs = 60000               # backstop caps (train-to-floor stops early)
regions = [:saddle, :stable_node, :unstable_node, :stable_spiral, :unstable_spiral, :center, :origin]

# canonical interior centers: disks of radius R_patch fit strictly inside each region.
# spirals are the binding constraint (they crowd the tau=0 axis and the parabola tau^2=4d).
# pole of inaccessibility within [-4.5,4.5]^2 (deepest interior point / largest inscribed disk)
centers = Dict(
    :saddle          => (F(0.0),  F(-2.25)),
    :stable_node     => (F(-3.45), F(1.05)),
    :unstable_node   => (F(3.45),  F(1.05)),
    :stable_spiral   => (F(-1.55), F(2.9)),
    :unstable_spiral => (F(1.55),  F(2.9)),
    :center          => (F(0.0),  F(2.25)),           # segment center (tau=0), degenerate
    :origin          => (F(0.0),  F(0.0)),             # disk at plane origin (mixes every regime)
)

# ---- extrapolation-sweep config ----
Lext = 8.0; shell_w = 0.4; n_ring = 600               # radial shells from each patch center (zoomed out)
shells = collect(shell_w/2 : shell_w : Lext)

# ---- placement-sensitivity config ----
place_K = 4                                           # random valid disks per region
place_box = F(3.5)                                    # window to sample candidate centers from
place_adam = 15000; place_lbfgs = 30000               # moderate (robustness, not floor)

# ---------------------------------------------------------------------------
# region classification + unified-eig helpers (shared with gen_radius.jl)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# FIXED-DISK patch: sunflower lattice (even, deterministic).  center -> 1-D segment.
# ---------------------------------------------------------------------------
const GA = F(π * (3 - sqrt(5)))                       # golden angle ~= 2.39996
function disk_at(cx, cy)                              # sunflower disk of radius R_patch at (cx,cy)
    ts = Vector{F}(undef, n_patch); ds = Vector{F}(undef, n_patch)
    for i in 1:n_patch
        r = R_patch * sqrt((F(i) - F(0.5)) / n_patch); θ = GA * i
        ts[i] = cx + r * cos(θ); ds[i] = cy + r * sin(θ)
    end
    return ts, ds
end
function patch_points(reg)
    cx, cy = centers[reg]
    if reg == :center
        ds = collect(range(cy - R_patch, cy + R_patch, length = n_patch))  # segment, same extent as disk radius
        return fill(F(0.0), n_patch), F.(ds)
    end
    return disk_at(cx, cy)
end
disk_in_region(ts, ds, reg) = all(region(ts[k], ds[k]) == reg for k in 1:length(ts))
function valid_center(reg, rng)                       # random center whose whole disk lies in `reg`
    while true
        cx = rand(rng, F) * 2place_box - place_box; cy = rand(rng, F) * 2place_box - place_box
        ts, ds = disk_at(cx, cy)
        disk_in_region(ts, ds, reg) && return cx, cy
    end
end

net = Lux.Chain(Lux.Dense(2, 64, tanh), Lux.Dense(64, 64, tanh), Lux.Dense(64, 64, tanh), Lux.Dense(64, 4))
_, st = Lux.setup(Random.default_rng(), net)

function batched_u(μ, k, A, B)                        # Mf×G, exact cosh/cos (Float64), overflow-safe
    G = length(μ); U = zeros(F, Mf, G)
    for g in 1:G
        kk = Float64(k[g]); mm = Float64(μ[g]); aa = Float64(A[g]); bb = Float64(B[g]); sk = sqrt(abs(kk))
        for m in 1:Mf
            x = Float64(xfine[m])
            C = kk >= 0 ? cosh(sk*x) : cos(sk*x)
            S = abs(kk) < 1e-12 ? x : (kk > 0 ? sinh(sk*x)/sk : sin(sk*x)/sk)
            val = exp(mm*x) * (aa*C + bb*S)
            U[m, g] = isfinite(val) ? F(val) : F(1e6)  # net overflowed -> mark as large error
        end
    end
    return U
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

# TRAIN TO A FLOOR: stop as soon as training loss < loss_floor (cap as backstop)
function train_to_floor(Xtr, Utr, ntr)
    p0, _ = Lux.setup(MersenneTwister(1234), net); p0ca = ComponentArray(p0)
    prob = OptimizationProblem(OptimizationFunction((p, _) -> loss_core(p, Xtr, Utr, ntr), Optimization.AutoZygote()), p0ca)
    stopcb = (s, l) -> l < loss_floor
    r = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); callback = stopcb, maxiters = patch_adam)
    padam = r.u; pfinal = padam                       # Adam result is finite; LBFGS may blow up far out
    try
        rl = solve(remake(prob; u0 = padam), OptimizationOptimJL.LBFGS(); callback = stopcb, maxiters = patch_lbfgs)
        isfinite(loss_core(rl.u, Xtr, Utr, ntr)) && (pfinal = rl.u)
    catch
    end
    return pfinal, Float64(loss_core(pfinal, Xtr, Utr, ntr))
end
function train_moderate(Xtr, Utr, ntr)               # fixed moderate budget (placement study)
    p0, _ = Lux.setup(MersenneTwister(1234), net); p0ca = ComponentArray(p0)
    prob = OptimizationProblem(OptimizationFunction((p, _) -> loss_core(p, Xtr, Utr, ntr), Optimization.AutoZygote()), p0ca)
    r = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); maxiters = place_adam)
    r = solve(remake(prob; u0 = r.u), OptimizationOptimJL.LBFGS(); maxiters = place_lbfgs)
    return r.u, Float64(loss_core(r.u, Xtr, Utr, ntr))
end

# trace-determinant region boundaries: Δ=0, τ=0 (Δ>0), and the parabola Δ=τ²/4
function overlay_regions!(pl)
    plot!(pl, [-Lmap, Lmap], [0.0, 0.0]; color = :white, lw = 1, ls = :dash, label = "")   # Δ=0
    plot!(pl, [0.0, 0.0], [0.0, Lmap];   color = :white, lw = 1, ls = :dash, label = "")   # τ=0, Δ>0
    tt = collect(range(-2*sqrt(Lmap), 2*sqrt(Lmap), length = 120))                         # parabola Δ=τ²/4
    plot!(pl, tt, (tt .^ 2) ./ 4; color = :white, lw = 1, ls = :dash, label = "")
    return pl
end

# ---------------------------------------------------------------------------
# shared test grid + true solution (batched once)
# ---------------------------------------------------------------------------
τg = collect(range(-Lmap, Lmap, length = Ng)); Δg = collect(range(-Lmap, Lmap, length = Ng))
Xgrid = reduce(hcat, [F[t, d] for d in Δg for t in τg]); Gpts = size(Xgrid, 2)
let μt = Xgrid[1:1, :] ./ 2, kt = Xgrid[1:1, :] .^ 2 ./ 4 .- Xgrid[2:2, :]
    global Utrue_grid = batched_u(μt, kt, fill(a0, 1, Gpts), a1 .- μt .* a0)
    global nrm_true = sum(abs2, Utrue_grid; dims = 1) .+ F(1e-12)
end
cell = (2Lmap / (Ng - 1))^2
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
# fair, geometry-free reach: r(theta) from the patch CENTER (same R_patch for all)
function reach_from_center(p, cx, cy)
    angles = collect(range(0, 2π, length = 121))[1:end-1]; step = 0.04; rs = Float64[]
    for θ in angles
        r = 0.0
        while r < Lext; pt_err(p, F(cx + r*cos(θ)), F(cy + r*sin(θ))) > eps_tol && break; r += step; end
        push!(rs, r)
    end
    return rs
end

# ===========================================================================
# Train one net per region (shared by both experiments)
# ===========================================================================
trained = Dict{Symbol,Any}()
println("training one net per region on identical R_patch=$R_patch disks (n=$n_patch), train-to-floor<$loss_floor")
for reg in regions
    ts, ds = patch_points(reg)
    frac_in = mean(region(ts[k], ds[k]) == reg for k in 1:n_patch)   # sanity: patch stays in-region
    Xtr = permutedims(hcat(ts, ds)); Utr = reduce(hcat, [utrue_vals(ts[k], ds[k], xs) for k in 1:n_patch])
    p, fl = train_to_floor(Xtr, Utr, n_patch)
    ok = fl < flag_thresh
    trained[reg] = (p = p, ts = ts, ds = ds, floor_reached = ok, finalloss = fl)
    println(rpad(string(reg), 16), "final train loss = ", round(fl; sigdigits = 3),
            ok ? "  (converged)" : "  (UNDERTRAINED: > $flag_thresh)",
            "   patch in-region = ", round(100*frac_in; digits = 1), "%")
end

# ===========================================================================
# [A] FAMILY MAPS: error over the plane + eps-contour + fair reach
# ===========================================================================
if run_maps
    panels = Plots.Plot[]
    println("\nregion            area(err<eps)   median_reach   margin(reach-R_patch)   |center|")
    for reg in regions
        tr = trained[reg]; cx, cy = centers[reg]; dorig = sqrt(Float64(cx)^2 + Float64(cy)^2)
        Emap, errs = plane_error_map(tr.p); area = sum(errs .< eps_tol) * cell
        rs = reach_from_center(tr.p, cx, cy); mreach = median(rs); margin = mreach - Float64(R_patch)
        println(rpad(string(reg), 16), rpad(round(area; sigdigits = 3), 15),
                rpad(round(mreach; sigdigits = 3), 14), rpad(round(margin; sigdigits = 3), 23),
                round(dorig; sigdigits = 3), tr.floor_reached ? "" : "   [undertrained]")
        pl = heatmap(τg, Δg, log10.(max.(Emap, 1e-4)); c = :viridis, clims = (-4, 0.5), colorbar = false,
                     title = "$reg\narea=$(round(area;sigdigits=3))  reach=$(round(mreach;sigdigits=2))",
                     titlefontsize = 8, xlabel = "τ", ylabel = "Δ",
                     aspect_ratio = :equal, xlims = (-Lmap, Lmap), ylims = (-Lmap, Lmap))
        contour!(pl, τg, Δg, Emap; levels = [eps_tol], color = :red, lw = 1.5, colorbar_entry = false)
        overlay_regions!(pl)
        scatter!(pl, tr.ts, tr.ds; ms = 1.6, color = :white, alpha = 0.6, markerstrokewidth = 0, label = "")
        push!(panels, pl)
    end
    fig = plot(panels...; layout = (2, 4), size = (2000, 950), left_margin = 5Plots.mm,
               bottom_margin = 7Plots.mm, top_margin = 4Plots.mm)
    savefig(fig, "data/family_disks_maps.png")
    println("Plot: data/family_disks_maps.png  (white dashed = region boundaries; white dots = training disk; red = ε)")
end

# ===========================================================================
# [B] OUT-OF-REGION EXTRAPOLATION: radial shells from each patch center
#     in-family (bold) vs out-of-family (dashed) vs out-of-range (rise with distance)
# ===========================================================================
if run_extrap
    panels = Plots.Plot[]
    for (idx, treg) in enumerate(regions)
        tr = trained[treg]; cx, cy = centers[treg]
        # accumulate per-(test region, shell) mean rel-L2 by binning ring points by region
        acc = Dict(r => [Float64[] for _ in shells] for r in regions)
        rng = MersenneTwister(2024)
        for (si, d) in enumerate(shells)
            for _ in 1:n_ring
                φ = 2π * rand(rng, F); t = cx + F(d)*cos(φ); dd = cy + F(d)*sin(φ)
                reg = region(t, dd)
                haskey(acc, reg) && push!(acc[reg][si], pt_err(tr.p, t, dd))
            end
        end
        p = plot(yscale = :log10, title = "train: $treg", xlabel = "distance from patch center",
                 ylabel = "mean rel-L2", titlefontsize = 10, legendfontsize = 6,
                 legend = (idx == 1 ? :bottomright : false))
        for (k, reg) in enumerate(regions)
            xs_r = Float64[]; ys_r = Float64[]
            for (si, d) in enumerate(shells)
                isempty(acc[reg][si]) && continue
                push!(xs_r, d); push!(ys_r, max(mean(acc[reg][si]), 1e-6))
            end
            isempty(xs_r) && continue
            is_tr = reg == treg
            plot!(p, xs_r, ys_r; color = k, marker = :circle, ms = 2,
                  lw = is_tr ? 3 : 1.2, ls = is_tr ? :solid : :dash, label = string(reg))
        end
        vline!(p, [Float64(R_patch)], color = :black, ls = :dot, label = "")   # patch edge
        hline!(p, [eps_tol], color = :red, ls = :dash, label = "")             # ε
        push!(panels, p)
    end
    fig = plot(panels...; layout = (2, 4), size = (2000, 900), left_margin = 5Plots.mm,
               bottom_margin = 8Plots.mm, top_margin = 4Plots.mm)
    savefig(fig, "data/family_disks_extrap.png")
    println("Plot: data/family_disks_extrap.png  (bold = in-family; dotted = patch edge; red = ε)")
end

# ===========================================================================
# [C] PLACEMENT SENSITIVITY: is the reach an artifact of WHERE the disk sits?
#     Train K random valid disks per region; report the spread of the reach margin.
#     If stable≻unstable holds across placements, the ranking is placement-robust.
# ===========================================================================
if run_placement
    regs2 = [r for r in regions if r != :center]      # center is degenerate (no 2-D disk)
    summary = Dict{Symbol,Vector{Float64}}()
    println("\nplacement sensitivity: $place_K random valid disks per region (moderate budget)")
    println("region            reach-margin: mean ± std        (per-disk values)")
    for (gi, reg) in enumerate(regs2)
        rng = MersenneTwister(100 + gi); margins = Float64[]
        for _ in 1:place_K
            cx, cy = valid_center(reg, rng)
            ts, ds = disk_at(cx, cy)
            Xtr = permutedims(hcat(ts, ds)); Utr = reduce(hcat, [utrue_vals(ts[k], ds[k], xs) for k in 1:n_patch])
            p, _ = train_moderate(Xtr, Utr, n_patch)
            rs = reach_from_center(p, cx, cy); push!(margins, median(rs) - Float64(R_patch))
        end
        summary[reg] = margins
        println(rpad(string(reg), 16), rpad("$(round(mean(margins);sigdigits=3)) ± $(round(std(margins);sigdigits=2))", 26),
                "  ", round.(margins; sigdigits = 3))
    end
    xpos = collect(1:length(regs2))
    means = [mean(summary[r]) for r in regs2]; stds = [std(summary[r]) for r in regs2]
    pp = bar(xpos, means; yerror = stds, legend = false, fillalpha = 0.5, color = :steelblue,
             xticks = (xpos, string.(regs2)), xrotation = 20, ylabel = "reach margin (reach − R_patch)",
             title = "placement sensitivity: mean ± std over $place_K random disks", bottom_margin = 12Plots.mm)
    for (i, r) in enumerate(regs2)
        scatter!(pp, fill(i, length(summary[r])) .+ 0.08 .* randn(length(summary[r])), summary[r];
                 color = :black, ms = 4, markerstrokewidth = 0, label = "")
    end
    savefig(pp, "data/family_disks_placement.png")
    println("Plot: data/family_disks_placement.png  (bars = mean reach margin, dots = individual disks)")
end
