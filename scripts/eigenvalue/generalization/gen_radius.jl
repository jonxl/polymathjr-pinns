#=
GENERALIZATION RADIUS  (two experiments, shared core, toggle at the top).

Unified-eig model, u = e^{mu x}(A C(k,x) + B S(k,x)), k = tau^2/4 - Delta (all regions).

  [1] DISK experiment (run_disk):  train on a compact disk of (tau,Delta); measure how far
      out the solution error stays below eps.
        - error map + eps-contour + training disk           -> data/gen_radius_map.png
        - radial profile r(theta) (min/median/max radius)   -> data/gen_radius_polar.png

  [2] PER-FAMILY maps (run_family):  train one model per trace-det region; map the error
      over the plane and draw the eps-contour ("generalization region"); its AREA is the
      score for that family.                                 -> data/family_radius_maps.png

OPTIMIZED: the plane error map is computed with BATCHED matrix ops (the net evaluates the
whole grid at once; u is reconstructed for all points via broadcasts), not a per-point loop.
=#

using Lux
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote, ComponentArrays
using Plots
import Random
using Random: MersenneTwister
using Statistics: median
using LinearAlgebra

isdir("data") || mkpath("data")
F = Float32
Random.seed!(1234)

# ---- what to run ----
run_disk   = false
run_family = true

# ---- shared config ----
a0 = F(1.0); a1 = F(0.0)
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)
x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))
xfine = collect(range(x_left, x_right, length = 200)); Mf = length(xfine)
eps_tol = 0.10
Lmap = 4.0; Ng = 81                                   # error-map half-width + resolution

# disk experiment
cτ, cΔ = F(0.0), F(0.0); R_train = F(1.5); n_disk = 600
disk_adam, disk_lbfgs = 20000, 50000                  # extensive: training a non-issue

# per-family experiment
tau_lim = F(2.0); delta_lim = F(2.0); n_fam = 500
fam_adam, fam_lbfgs = 20000, 50000                   # extensive (same as disk), per region
regions = [:saddle, :stable_node, :unstable_node, :stable_spiral, :unstable_spiral, :center]

# ---------------------------------------------------------------------------
# Unified representation helpers
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
function sample_region(reg, n, rng)
    ts = F[]; ds = F[]
    if reg == :center
        while length(ts) < n; push!(ts, F(0.0)); push!(ds, rand(rng, F) * delta_lim); end
    else
        while length(ts) < n
            t = rand(rng, F)*(2tau_lim) - tau_lim; d = rand(rng, F)*(2delta_lim) - delta_lim
            region(t, d) == reg && (push!(ts, t); push!(ds, d))
        end
    end
    return ts, ds
end

net = Lux.Chain(Lux.Dense(2, 64, tanh), Lux.Dense(64, 64, tanh), Lux.Dense(64, 64, tanh), Lux.Dense(64, 4))
_, st = Lux.setup(Random.default_rng(), net)

# batched u on xfine for a whole batch of params (each 1×G) -> Mf×G
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
function train(Xtr, Utr, ntr; ai, li)
    p0, _ = Lux.setup(MersenneTwister(1234), net); p0ca = ComponentArray(p0)
    prob = OptimizationProblem(OptimizationFunction((p, _) -> loss_core(p, Xtr, Utr, ntr), Optimization.AutoZygote()), p0ca)
    r = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); maxiters = ai)
    r = solve(remake(prob; u0 = r.u), OptimizationOptimJL.LBFGS(); maxiters = li)
    return r.u
end

# shared test grid + true solution on it (batched, once)
τg = collect(range(-Lmap, Lmap, length = Ng)); Δg = collect(range(-Lmap, Lmap, length = Ng))
Xgrid = reduce(hcat, [F[t, d] for d in Δg for t in τg])           # 2×G, Δ outer / τ inner
Gpts = size(Xgrid, 2)
let μt = Xgrid[1:1, :] ./ 2, kt = Xgrid[1:1, :] .^ 2 ./ 4 .- Xgrid[2:2, :]
    global Utrue_grid = batched_u(μt, kt, fill(a0, 1, Gpts), a1 .- μt .* a0)
    global nrm_true = sum(abs2, Utrue_grid; dims = 1) .+ F(1e-12)
end
cell = (2Lmap / (Ng - 1))^2
function plane_error_map(p)                                       # -> Ng×Ng matrix [Δ,τ]
    O = first(net(Xgrid, p, st))
    Upred = batched_u(O[1:1, :], O[2:2, :], O[3:3, :], O[4:4, :])
    errs = vec(sqrt.(sum(abs2, Upred .- Utrue_grid; dims = 1) ./ nrm_true))
    return permutedims(reshape(errs, Ng, Ng)), errs
end
function pt_err(p, t, d)                                          # scalar (for radial march)
    O = vec(first(net(reshape(F[t, d], 2, 1), p, st))); μ, k, A, B = O[1], O[2], O[3], O[4]
    up = [exp(μ*x)*(A*Cfun(k, x) + B*Sfun(k, x)) for x in xfine]; ut = utrue_vals(t, d, xfine)
    sqrt(sum(abs2, up .- ut) / (sum(abs2, ut) + 1e-12))
end

# ===========================================================================
# [1] DISK experiment
# ===========================================================================
if run_disk
    taus = F[]; deltas = F[]
    for _ in 1:n_disk
        ρ = R_train*sqrt(rand(F)); φ = 2π*rand(F); push!(taus, cτ + ρ*cos(φ)); push!(deltas, cΔ + ρ*sin(φ))
    end
    Xtr = permutedims(hcat(taus, deltas)); Utr = reduce(hcat, [utrue_vals(taus[k], deltas[k], xs) for k in 1:n_disk])
    p_disk = train(Xtr, Utr, n_disk; ai = disk_adam, li = disk_lbfgs)
    println("disk trained  center=($cτ,$cΔ) R_train=$R_train")

    Emap, _ = plane_error_map(p_disk)
    hm = heatmap(τg, Δg, log10.(max.(Emap, 1e-4)); c = :viridis, colorbar_title = "log10 rel-L2",
                 xlabel = "τ", ylabel = "Δ", title = "disk error map + ε=$eps_tol contour")
    contour!(hm, τg, Δg, Emap; levels = [eps_tol], color = :red, lw = 2, colorbar_entry = false)
    θc = range(0, 2π, length = 200)
    plot!(hm, cτ .+ R_train .* cos.(θc), cΔ .+ R_train .* sin.(θc); color = :white, ls = :dash, lw = 2, label = "train disk")
    savefig(hm, "data/gen_radius_map.png")

    angles = collect(range(0, 2π, length = 145))[1:end-1]; step = 0.04; rs = Float64[]
    for θ in angles
        r = 0.0
        while r < Lmap; pt_err(p_disk, F(cτ + r*cos(θ)), F(cΔ + r*sin(θ))) > eps_tol && break; r += step; end
        push!(rs, r)
    end
    println("ε=$eps_tol R_train=$R_train  radius r(θ): min=$(round(minimum(rs);sigdigits=3)) ",
            "median=$(round(median(rs);sigdigits=3)) max=$(round(maximum(rs);sigdigits=3))  ",
            "median margin=$(round(median(rs)-R_train;sigdigits=3))")
    prad = plot(angles, rs; proj = :polar, lw = 2, label = "r(θ)", title = "generalization radius r(θ), ε=$eps_tol")
    plot!(prad, angles, fill(Float64(R_train), length(angles)); proj = :polar, lw = 2, ls = :dash, color = :gray, label = "train radius")
    savefig(prad, "data/gen_radius_polar.png")
    println("Plots: data/gen_radius_map.png , data/gen_radius_polar.png")
end

# ===========================================================================
# [2] PER-FAMILY maps
# ===========================================================================
if run_family
    panels = Plots.Plot[]; areas = Float64[]
    for reg in regions
        rng = MersenneTwister(1234); ts, ds = sample_region(reg, n_fam, rng)
        Xtr = permutedims(hcat(ts, ds)); Utr = reduce(hcat, [utrue_vals(ts[k], ds[k], xs) for k in 1:n_fam])
        p = train(Xtr, Utr, n_fam; ai = fam_adam, li = fam_lbfgs)
        Emap, errs = plane_error_map(p); area = sum(errs .< eps_tol) * cell
        push!(areas, area); println("trained on $reg   area(err<ε) = $(round(area; sigdigits = 3))")
        pl = heatmap(τg, Δg, log10.(max.(Emap, 1e-4)); c = :viridis, clims = (-4, 0.5), colorbar = false,
                     title = "$reg   area=$(round(area; sigdigits = 3))", titlefontsize = 8, xlabel = "τ", ylabel = "Δ")
        contour!(pl, τg, Δg, Emap; levels = [eps_tol], color = :red, lw = 1.5, colorbar_entry = false)
        scatter!(pl, ts, ds; ms = 1.3, color = :white, alpha = 0.3, markerstrokewidth = 0, label = "")
        push!(panels, pl)
    end
    println("\nfamily         area(err<ε=$eps_tol)")
    for k in eachindex(regions); println(rpad(string(regions[k]), 16), round(areas[k]; sigdigits = 3)); end
    fig = plot(panels...; layout = (2, 3), size = (1500, 900))
    savefig(fig, "data/family_radius_maps.png")
    println("Plot: data/family_radius_maps.png  (red = ε-contour, white dots = training region)")
end
