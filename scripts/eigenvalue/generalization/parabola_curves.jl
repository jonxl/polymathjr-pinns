#=
PER-NET error curves along the parabola  Delta = tau^2/4.  For each of several TINY training
segments (a fresh net each), train on that segment and test ALONG the whole parabola; save an
INDIVIDUAL plot of rel-L2 vs position (tau) on the curve.

Each plot: x = tau along the parabola (Delta = tau^2/4 implied), y = solution rel-L2 (log),
green band = the training segment, red dashed = eps.  So you see the error valley at the
segment and how it grows as you move along the curve to either side.

Unified-eig model; metric = solution-space relative L2.
Outputs: data/parab_curve_01.png ... (one per net) + data/parab_curves_all.png (contact sheet).
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

# ---- config ----
a0 = F(1.0); a1 = F(0.0)
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)
x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))
xfine = collect(range(x_left, x_right, length = 200)); Mf = length(xfine)
eps_tol = 0.10

w = F(0.3)                                             # TINY training segment: tau in [tc-w, tc+w]
n_train = 80
centers = Float64.([-3.0, -2.25, -1.5, -0.75, 0.0, 0.75, 1.5, 2.25, 3.0])   # one fresh net each
Ltau = 20.0; n_test = 800                              # test along parabola tau in [-Ltau, Ltau] (extended range)
loss_floor = F(1e-6); seg_adam = 20000; seg_lbfgs = 40000

# ---- unified-eig helpers ----
const Pterm = 14
sxE = [F.(xs    .^ (2n))     ./ F(factorial(big(2n)))     for n in 0:Pterm]
sxO = [F.(xs    .^ (2n + 1)) ./ F(factorial(big(2n + 1))) for n in 0:Pterm]
CS_xs(k) = (sum(sxE[n+1] .* (k .^ n) for n in 0:Pterm), sum(sxO[n+1] .* (k .^ n) for n in 0:Pterm))
Cfun(k, x) = k >= 0 ? cosh(sqrt(k) * x) : cos(sqrt(-k) * x)
Sfun(k, x) = abs(k) < 1e-12 ? x : (k > 0 ? sinh(sqrt(k) * x) / sqrt(k) : sin(sqrt(-k) * x) / sqrt(-k))
function usol(μ, k, A, B, x)                            # numerically stable (no giant cosh) for extended τ
    if k > 1e-12
        s = sqrt(k); return (A/2 + B/(2s))*exp((μ+s)*x) + (A/2 - B/(2s))*exp((μ-s)*x)
    elseif k < -1e-12
        ω = sqrt(-k); return exp(μ*x)*(A*cos(ω*x) + B*sin(ω*x)/ω)
    else
        return exp(μ*x)*(A + B*x)
    end
end
function utrue_vals(t, d, xpts)
    μ = Float64(t)/2; k = Float64(t)^2/4 - Float64(d); A = Float64(a0); B = Float64(a1) - μ*Float64(a0)
    return F[ usol(μ, k, A, B, Float64(x)) for x in xpts ]
end

net = Lux.Chain(Lux.Dense(2, 64, tanh), Lux.Dense(64, 64, tanh), Lux.Dense(64, 64, tanh), Lux.Dense(64, 4))
_, st = Lux.setup(Random.default_rng(), net)
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
    r = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); callback = stopcb, maxiters = seg_adam)
    r = solve(remake(prob; u0 = r.u), OptimizationOptimJL.LBFGS(); callback = stopcb, maxiters = seg_lbfgs)
    return r.u, Float64(loss_core(r.u, Xtr, Utr, ntr))
end
function pt_err(p, t, d)
    O = vec(first(net(reshape(F[t, d], 2, 1), p, st))); μ, k, A, B = Float64(O[1]), Float64(O[2]), Float64(O[3]), Float64(O[4])
    up = [usol(μ, k, A, B, x) for x in Float64.(xfine)]; ut = utrue_vals(t, d, xfine)
    e = sqrt(sum(abs2, up .- ut) / (sum(abs2, ut) + 1e-12)); return isfinite(e) ? e : 1e3
end
err_parab(p, t) = pt_err(p, F(t), F(t^2 / 4))

τtest = collect(range(-Ltau, Ltau, length = n_test))

# ===========================================================================
# one fresh net per tiny segment -> individual error-along-parabola plot
# ===========================================================================
panels = Plots.Plot[]
for (idx, τc) in enumerate(centers)
    tt = collect(range(τc - Float64(w), τc + Float64(w), length = n_train))
    ts = F.(tt); ds = F.((tt .^ 2) ./ 4)
    Xtr = permutedims(hcat(ts, ds)); Utr = reduce(hcat, [utrue_vals(ts[k], ds[k], xs) for k in 1:n_train])
    p, fl = train_to_floor(Xtr, Utr, n_train)
    errs = [err_parab(p, t) for t in τtest]
    println("net $idx  τc=$τc  train loss=", round(fl; sigdigits = 3),
            fl < F(1e-5) ? "" : "  (UNDERTRAINED)")

    pl = plot(τtest, max.(errs, 1e-6); yscale = :log10, lw = 2.2, color = :purple, legend = false,
              xlabel = "τ along parabola  (Δ = τ²/4)", ylabel = "rel-L2 (log)",
              title = "net $idx: trained on τ∈[$(round(τc-Float64(w);digits=2)), $(round(τc+Float64(w);digits=2))]",
              titlefontsize = 9, ylims = (1e-6, 5))
    vspan!(pl, [τc - Float64(w), τc + Float64(w)]; color = :green, alpha = 0.18, label = "")
    hline!(pl, [eps_tol]; color = :red, ls = :dash, label = "")
    vline!(pl, [0.0]; color = :gray, ls = :dot, label = "")       # vertex: left=stable, right=unstable
    savefig(pl, "data/parab_curve_$(lpad(idx, 2, '0')).png")
    push!(panels, pl)
end
savefig(plot(panels...; layout = (3, 3), size = (1500, 1050)), "data/parab_curves_all.png")
println("\nSaved: data/parab_curve_01.png .. data/parab_curve_$(lpad(length(centers),2,'0')).png  + data/parab_curves_all.png")
