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
run_disk   = true
run_family = false
run_ic     = false         # Task 1: fix the ODE, vary the initial conditions

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

# IC-axis experiment (Task 1): fix the ODE, vary the initial conditions (y0,y0') on a disk
τ0_ic, Δ0_ic = F(0.0), F(-1.0)                        # fixed ODE (saddle: roots ±1)
cy0, cyp = F(0.0), F(0.0); R_ic = F(1.5); n_ic = 600  # IC disk (mirrors the ODE disk)
ic_adam, ic_lbfgs = 20000, 50000

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

# raw pde/ic/sup loss components for a batch. tv,dv = (τ,Δ) rows; y0v,ypv = IC-target rows.
function comps(p, Xb, Ub, nb, tv, dv, y0v, ypv)
    O = first(net(Xb, p, st)); μ = O[1:1, :]; k = O[2:2, :]; A = O[3:3, :]; B = O[4:4, :]
    C, S = CS_xs(k); v = A .* C .+ B .* S; vp = A .* (k .* S) .+ B .* C; E = exp.(xs * μ); U = E .* v
    lp = sum(abs2, E .* ((μ .^ 2 .+ k .- tv .* μ .+ dv) .* v .+ (2 .* μ .- tv) .* vp)) / (n_colloc * nb)
    li = (sum(abs2, A .- y0v) + sum(abs2, (μ .* A .+ B) .- ypv)) / nb
    ls = sum(abs2, U .- Ub) / (n_colloc * nb)
    return (lp, li, ls)
end
showcomps(tag, c) = println("  $tag loss components: pde=$(round(c[1];sigdigits=3)) ic=$(round(c[2];sigdigits=3)) sup=$(round(c[3];sigdigits=3))")

# ===========================================================================
# [1] DISK experiment
# ===========================================================================
if run_disk
    taus = F[]; deltas = F[]
    for _ in 1:n_disk
        ρ = R_train*sqrt(rand(F)); φ = 2π*rand(F); push!(taus, cτ + ρ*cos(φ)); push!(deltas, cΔ + ρ*sin(φ))
    end
    Xtr = permutedims(hcat(taus, deltas)); Utr = reduce(hcat, [utrue_vals(taus[k], deltas[k], xs) for k in 1:n_disk])
    hp = Float64[]; hi = Float64[]; hs = Float64[]; cnt = Ref(0)      # log training loss components every 20 iters
    cbd = function (s, l)
        cnt[] += 1
        if cnt[] % 20 == 0
            c = comps(s.u, Xtr, Utr, n_disk, Xtr[1:1, :], Xtr[2:2, :], a0, a1)
            push!(hp, c[1]); push!(hi, c[2]); push!(hs, c[3])
        end
        return false
    end
    p0d, _ = Lux.setup(MersenneTwister(1234), net)
    probd = OptimizationProblem(OptimizationFunction((p, _) -> loss_core(p, Xtr, Utr, n_disk), Optimization.AutoZygote()), ComponentArray(p0d))
    rd = solve(probd, OptimizationOptimisers.Adam(F(1e-3)); callback = cbd, maxiters = disk_adam); n_adam_d = length(hp)
    rd = solve(remake(probd; u0 = rd.u), OptimizationOptimJL.LBFGS(); callback = cbd, maxiters = disk_lbfgs)
    p_disk = rd.u
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

    # 3-tier + solution curves (pred vs true) -- direct "how good" view
    memerr = pt_err(p_disk, taus[1], deltas[1])
    rngi = MersenneTwister(77); interr = Float64[]
    for _ in 1:200
        ρ = R_train*sqrt(rand(rngi, F)); φ = 2π*rand(rngi, F); push!(interr, pt_err(p_disk, F(cτ+ρ*cos(φ)), F(cΔ+ρ*sin(φ))))
    end
    println("  memorization rel-L2 = $(round(memerr;sigdigits=3))   interpolation median = $(round(median(interr);sigdigits=3))")
    ctr = comps(p_disk, Xtr, Utr, n_disk, Xtr[1:1, :], Xtr[2:2, :], a0, a1); showcomps("train ", ctr)
    tex = F[]; dex = F[]; rnge = MersenneTwister(55)
    while length(tex) < 300
        ρ = R_train + rand(rnge, F)*(F(Lmap)-R_train); φ = 2π*rand(rnge, F); push!(tex, cτ+ρ*cos(φ)); push!(dex, cΔ+ρ*sin(φ))
    end
    nex = length(tex); Xex = permutedims(hcat(tex, dex)); Uex = reduce(hcat, [utrue_vals(tex[k], dex[k], xs) for k in 1:nex])
    cex = comps(p_disk, Xex, Uex, nex, Xex[1:1, :], Xex[2:2, :], a0, a1); showcomps("extrap", cex)
    # one png: (left) which component is high at extrapolation ; (right) training loss curves
    xp3 = [1, 2, 3]
    p1 = bar(xp3, max.(collect(cex), 1e-12); yscale = :log10, xticks = (xp3, ["pde", "ic", "sup"]), legend = false,
             color = :firebrick, ylabel = "loss component (log)", title = "disk (ODE): loss components at extrapolation")
    its = (1:length(hp)) .* 20
    p2 = plot(its, max.(hp, 1e-12); yscale = :log10, lw = 2, label = "pde", xlabel = "iteration (Adam→LBFGS)",
              ylabel = "training loss component", title = "disk (ODE): training loss curves")
    plot!(p2, its, max.(hi, 1e-12); lw = 2, label = "ic"); plot!(p2, its, max.(hs, 1e-12); lw = 2, label = "sup")
    vline!(p2, [n_adam_d*20 + 0.5]; ls = :dash, color = :gray, label = "Adam|LBFGS")
    savefig(plot(p1, p2; layout = (1, 2), size = (1300, 500)), "data/gen_radius_components.png")
    ode_samples = [(Float64(taus[1]), Float64(deltas[1]), "memorization"), (0.0, -1.0, "in-disk"),
                   (2.5, -1.0, "extrapolated"), (-2.5, 1.5, "extrapolated")]
    cpanels = Plots.Plot[]
    for (t, d, tag) in ode_samples
        O = vec(first(net(reshape(F[t, d], 2, 1), p_disk, st))); μ, k, A, B = O[1], O[2], O[3], O[4]
        up = [exp(μ*x)*(A*Cfun(k, x) + B*Sfun(k, x)) for x in xfine]; ut = utrue_vals(t, d, xfine)
        e = sqrt(sum(abs2, up .- ut)/(sum(abs2, ut)+1e-12))
        pl = plot(xfine, ut, lw = 3, color = :black, ls = :dash, label = "true",
                  title = "(τ,Δ)=($t,$d)  $tag\nrel-L2=$(round(e;sigdigits=2))", xlabel = "x", ylabel = "u(x)",
                  legend = :best, titlefontsize = 8)
        plot!(pl, xfine, up, lw = 2, color = :dodgerblue, label = "pred"); push!(cpanels, pl)
    end
    savefig(plot(cpanels...; layout = (2, 2), size = (1000, 720)), "data/gen_radius_curves.png")
    println("Plots: data/gen_radius_map.png , data/gen_radius_polar.png , data/gen_radius_curves.png")
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

# ===========================================================================
# [3] IC-AXIS experiment (Task 1): fix the ODE, vary the initial conditions
# ===========================================================================
if run_ic
    μ0 = Float64(τ0_ic) / 2; k0 = Float64(τ0_ic)^2 / 4 - Float64(Δ0_ic)   # fixed ODE -> fixed μ,k
    utrue_ic(y0, yp, xpts) = begin
        A = Float64(y0); B = Float64(yp) - μ0 * Float64(y0)
        F[ exp(μ0 * x) * (A * Cfun(k0, x) + B * Sfun(k0, x)) for x in Float64.(xpts) ]
    end
    rng = MersenneTwister(1234); ys = F[]; yps = F[]
    for _ in 1:n_ic
        ρ = R_ic * sqrt(rand(rng, F)); φ = 2π * rand(rng, F)
        push!(ys, cy0 + ρ * cos(φ)); push!(yps, cyp + ρ * sin(φ))
    end
    Xic = permutedims(hcat(ys, yps))
    Uic = reduce(hcat, [utrue_ic(ys[k], yps[k], xs) for k in 1:n_ic])
    # residual uses the FIXED ODE (τ0_ic,Δ0_ic); IC term targets the VARYING (y0,y0')
    function loss_ic(p, _)
        O = first(net(Xic, p, st)); μ = O[1:1, :]; k = O[2:2, :]; A = O[3:3, :]; B = O[4:4, :]
        y0 = Xic[1:1, :]; yp = Xic[2:2, :]
        C, S = CS_xs(k); v = A .* C .+ B .* S; vp = A .* (k .* S) .+ B .* C; E = exp.(xs * μ); U = E .* v
        resid = E .* ((μ .^ 2 .+ k .- τ0_ic .* μ .+ Δ0_ic) .* v .+ (2 .* μ .- τ0_ic) .* vp)
        lp = sum(abs2, resid) / (n_colloc * n_ic)
        li = (sum(abs2, A .- y0) + sum(abs2, (μ .* A .+ B) .- yp)) / n_ic
        ls = sum(abs2, U .- Uic) / (n_colloc * n_ic)
        return pde_weight * lp + ic_weight * li + sup_weight * ls
    end
    hpi = Float64[]; hii = Float64[]; hsi = Float64[]; cnti = Ref(0)   # log training loss components every 20 iters
    cbi = function (s, l)
        cnti[] += 1
        if cnti[] % 20 == 0
            c = comps(s.u, Xic, Uic, n_ic, τ0_ic, Δ0_ic, Xic[1:1, :], Xic[2:2, :])
            push!(hpi, c[1]); push!(hii, c[2]); push!(hsi, c[3])
        end
        return false
    end
    p0, _ = Lux.setup(MersenneTwister(1234), net); p0ca = ComponentArray(p0)
    prob = OptimizationProblem(OptimizationFunction(loss_ic, Optimization.AutoZygote()), p0ca)
    r = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); callback = cbi, maxiters = ic_adam); n_adam_i = length(hpi)
    r = solve(remake(prob; u0 = r.u), OptimizationOptimJL.LBFGS(); callback = cbi, maxiters = ic_lbfgs)
    p_ic = r.u
    println("IC axis trained: fixed ODE (τ,Δ)=($τ0_ic,$Δ0_ic) -> u = y0·cosh + y0'·sinh; disk R=$R_ic")

    function pt_err_ic(y0, yp)
        O = vec(first(net(reshape(F[y0, yp], 2, 1), p_ic, st))); μ, k, A, B = O[1], O[2], O[3], O[4]
        up = [exp(μ * x) * (A * Cfun(k, x) + B * Sfun(k, x)) for x in xfine]
        sqrt(sum(abs2, up .- utrue_ic(y0, yp, xfine)) / (sum(abs2, utrue_ic(y0, yp, xfine)) + 1e-12))
    end
    # shared 3-tier
    memerr = pt_err_ic(ys[1], yps[1])
    rngi = MersenneTwister(77); interr = Float64[]
    for _ in 1:200
        ρ = R_ic * sqrt(rand(rngi, F)); φ = 2π * rand(rngi, F); push!(interr, pt_err_ic(cy0 + ρ*cos(φ), cyp + ρ*sin(φ)))
    end
    angles = collect(range(0, 2π, length = 145))[1:end-1]; step = 0.04; rs = Float64[]
    for θ in angles
        rr = 0.0; while rr < Lmap; pt_err_ic(F(cy0 + rr*cos(θ)), F(cyp + rr*sin(θ))) > eps_tol && break; rr += step; end
        push!(rs, rr)
    end
    println("  memorization         rel-L2 = $(round(memerr; sigdigits = 3))")
    println("  interpolation median rel-L2 = $(round(median(interr); sigdigits = 3))")
    println("  IC radius r(θ): min=$(round(minimum(rs);sigdigits=3)) median=$(round(median(rs);sigdigits=3)) max=$(round(maximum(rs);sigdigits=3))  (R_train=$R_ic)")
    ctr = comps(p_ic, Xic, Uic, n_ic, τ0_ic, Δ0_ic, Xic[1:1, :], Xic[2:2, :]); showcomps("train ", ctr)
    yex = F[]; ypex = F[]; rnge = MersenneTwister(55)
    while length(yex) < 300
        ρ = R_ic + rand(rnge, F)*(F(Lmap)-R_ic); φ = 2π*rand(rnge, F); push!(yex, cy0+ρ*cos(φ)); push!(ypex, cyp+ρ*sin(φ))
    end
    nex = length(yex); Xex = permutedims(hcat(yex, ypex)); Uex = reduce(hcat, [utrue_ic(yex[k], ypex[k], xs) for k in 1:nex])
    cex = comps(p_ic, Xex, Uex, nex, τ0_ic, Δ0_ic, Xex[1:1, :], Xex[2:2, :]); showcomps("extrap", cex)
    # one png: (left) which component is high at extrapolation ; (right) training loss curves
    xp3 = [1, 2, 3]
    p1 = bar(xp3, max.(collect(cex), 1e-12); yscale = :log10, xticks = (xp3, ["pde", "ic", "sup"]), legend = false,
             color = :firebrick, ylabel = "loss component (log)", title = "IC axis: loss components at extrapolation")
    its = (1:length(hpi)) .* 20
    p2 = plot(its, max.(hpi, 1e-12); yscale = :log10, lw = 2, label = "pde", xlabel = "iteration (Adam→LBFGS)",
              ylabel = "training loss component", title = "IC axis: training loss curves")
    plot!(p2, its, max.(hii, 1e-12); lw = 2, label = "ic"); plot!(p2, its, max.(hsi, 1e-12); lw = 2, label = "sup")
    vline!(p2, [n_adam_i*20 + 0.5]; ls = :dash, color = :gray, label = "Adam|LBFGS")
    savefig(plot(p1, p2; layout = (1, 2), size = (1300, 500)), "data/gen_radius_ic_components.png")

    # error map over the (y0,y0') plane (reuse the τg/Δg grid ranges = [-Lmap,Lmap])
    Xg = reduce(hcat, [F[a, b] for b in Δg for a in τg])
    Og = first(net(Xg, p_ic, st))
    Upred = batched_u(Og[1:1, :], Og[2:2, :], Og[3:3, :], Og[4:4, :])
    y0r = Xg[1:1, :]; ypr = Xg[2:2, :]
    Utr = batched_u(fill(F(μ0), 1, Gpts), fill(F(k0), 1, Gpts), y0r, ypr .- F(μ0) .* y0r)
    errs = vec(sqrt.(sum(abs2, Upred .- Utr; dims = 1) ./ (sum(abs2, Utr; dims = 1) .+ F(1e-12))))
    Emap = permutedims(reshape(errs, Ng, Ng))
    hm = heatmap(τg, Δg, log10.(max.(Emap, 1e-5)); c = :viridis, colorbar_title = "log10 rel-L2",
                 xlabel = "y₀", ylabel = "y₀'", title = "IC axis (fixed ODE): error over (y₀,y₀'), ε=$eps_tol")
    contour!(hm, τg, Δg, Emap; levels = [eps_tol], color = :red, lw = 2, colorbar_entry = false)
    θc = range(0, 2π, length = 200)
    plot!(hm, cy0 .+ R_ic .* cos.(θc), cyp .+ R_ic .* sin.(θc); color = :white, ls = :dash, lw = 2, label = "train disk")
    savefig(hm, "data/gen_radius_ic_map.png")
    prad = plot(angles, rs; proj = :polar, lw = 2, label = "r(θ)", title = "IC-axis radius (fixed ODE), ε=$eps_tol")
    plot!(prad, angles, fill(Float64(R_ic), length(angles)); proj = :polar, ls = :dash, color = :gray, label = "train radius")
    savefig(prad, "data/gen_radius_ic_polar.png")

    # solution curves (pred vs true) at sample ICs -- direct "how good" view
    ic_samples = [(Float64(ys[1]), Float64(yps[1]), "memorization"), (0.5, 0.5, "in-disk"),
                  (2.0, 2.0, "extrapolated (growing)"), (2.0, -2.0, "extrapolated (decaying)")]
    cpanels = Plots.Plot[]
    for (y0, yp, tag) in ic_samples
        O = vec(first(net(reshape(F[y0, yp], 2, 1), p_ic, st))); μ, k, A, B = O[1], O[2], O[3], O[4]
        up = [exp(μ*x)*(A*Cfun(k, x) + B*Sfun(k, x)) for x in xfine]; ut = utrue_ic(y0, yp, xfine)
        e = sqrt(sum(abs2, up .- ut)/(sum(abs2, ut)+1e-12))
        pl = plot(xfine, ut, lw = 3, color = :black, ls = :dash, label = "true",
                  title = "(y₀,y₀')=($y0,$yp)  $tag\nrel-L2=$(round(e;sigdigits=2))", xlabel = "x", ylabel = "u(x)",
                  legend = :best, titlefontsize = 8)
        plot!(pl, xfine, up, lw = 2, color = :dodgerblue, label = "pred"); push!(cpanels, pl)
    end
    savefig(plot(cpanels...; layout = (2, 2), size = (1000, 720)), "data/gen_radius_ic_curves.png")
    println("Plots: data/gen_radius_ic_map.png , data/gen_radius_ic_polar.png , data/gen_radius_ic_curves.png")
end
