#=
SLIDING SEGMENT along the parabola  Delta = tau^2/4.  Fixed-size window, stepped outward from
the vertex up BOTH arms (tau<0 stable, tau>0 unstable), a FRESH net trained per position.
Question: how does generalization depend on WHERE on the parabola you train, and is the
stable/unstable asymmetry a function of position?

For each window (center tau_c, half-width w) we train on parabola points in [tau_c-w, tau_c+w],
then:
  * test ALONG the whole parabola -> an error curve (row of a heatmap),
  * measure OUTWARD reach (march to larger |tau|, toward more extreme dynamics) and INWARD reach
    (march toward the vertex) before rel-L2 crosses eps.
Unified-eig model; metric = solution-space relative L2.
Outputs: data/parabola_slide_maps.png  (per-arm heatmaps), data/parabola_slide_reach.png (summary)
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

w = F(0.75)                                   # window half-width in tau (segment width = 2w = 1.5)
centers_pos = F.([0.75, 1.5, 2.25, 3.0, 3.75])  # unstable-arm window centers (tiles vertex->4.5)
n_win_train = 200
Ltau = 4.5; n_test = 300                       # test along parabola tau in [-Ltau, Ltau]
loss_floor = F(1e-6); patch_adam = 30000; patch_lbfgs = 60000

# ---- region + unified-eig helpers ----
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
    r = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); callback = stopcb, maxiters = patch_adam)
    r = solve(remake(prob; u0 = r.u), OptimizationOptimJL.LBFGS(); callback = stopcb, maxiters = patch_lbfgs)
    return r.u, Float64(loss_core(r.u, Xtr, Utr, ntr))
end
function pt_err(p, t, d)
    O = vec(first(net(reshape(F[t, d], 2, 1), p, st))); μ, k, A, B = O[1], O[2], O[3], O[4]
    up = [exp(μ*x)*(A*Cfun(k, x) + B*Sfun(k, x)) for x in xfine]; ut = utrue_vals(t, d, xfine)
    sqrt(sum(abs2, up .- ut) / (sum(abs2, ut) + 1e-12))
end
err_parab(p, t) = pt_err(p, F(t), F(t^2 / 4))         # error at parabola point with abscissa tau=t
function reach_along(p, τstart, dir; step = 0.03, maxd = 2.5)   # march along parabola until err>eps
    d = 0.0
    while d < maxd
        err_parab(p, τstart + dir * (d + step)) > eps_tol && break
        d += step
    end
    return d
end

τtest = collect(range(-Ltau, Ltau, length = n_test))

# train one window, return (net params, error curve over τtest, in-seg median, outward & inward reach)
function run_window(τc)
    tt = collect(range(τc - Float64(w), τc + Float64(w), length = n_win_train))
    ts = F.(tt); ds = F.((tt .^ 2) ./ 4)
    Xtr = permutedims(hcat(ts, ds)); Utr = reduce(hcat, [utrue_vals(ts[k], ds[k], xs) for k in 1:n_win_train])
    p, fl = train_to_floor(Xtr, Utr, n_win_train)
    curve = [err_parab(p, t) for t in τtest]
    inseg = median([err_parab(p, t) for t in tt])
    outer = τc + sign(τc) * Float64(w); inner = τc - sign(τc) * Float64(w)   # edges (|outer|>|inner|)
    r_out = reach_along(p, outer, sign(τc))          # toward larger |τ| (more extreme dynamics)
    r_in  = reach_along(p, inner, -sign(τc))         # toward the vertex
    return (curve = curve, inseg = inseg, r_out = r_out, r_in = r_in, converged = fl < F(1e-5))
end

# ===========================================================================
# sweep both arms
# ===========================================================================
centers = vcat(-reverse(centers_pos), centers_pos)   # stable arm (neg) then unstable arm (pos)
E = zeros(length(centers), n_test)
insegs = Float64[]; r_outs = Float64[]; r_ins = Float64[]
println("sliding window w=$w along both parabola arms; fresh net per position (train-to-floor<$loss_floor)")
println("τc        in-seg      reach_out   reach_in   conv")
for (i, τc) in enumerate(centers)
    res = run_window(Float64(τc))
    E[i, :] = res.curve; push!(insegs, res.inseg); push!(r_outs, res.r_out); push!(r_ins, res.r_in)
    println(rpad(round(τc; digits = 2), 10), rpad(round(res.inseg; sigdigits = 3), 12),
            rpad(round(res.r_out; sigdigits = 3), 12), rpad(round(res.r_in; sigdigits = 3), 11),
            res.converged ? "yes" : "NO")
end

# ---------------------------------------------------------------------------
# (τ,Δ)-PLANE view: each training window is an ARC on the parabola; thin whiskers show how far
# it generalizes along the curve before rel-L2 crosses ε (solid = outward toward the extreme,
# dashed = inward toward the vertex).  blue = stable arm, red = unstable arm.
# ---------------------------------------------------------------------------
arc(a, b; n = 80) = (t = collect(range(a, b, length = n)); (t, (t .^ 2) ./ 4))
pnrm(t) = (v = (-t/2, 1.0); m = sqrt(v[1]^2 + v[2]^2); (v[1]/m, v[2]/m))   # unit normal to parabola
pbx = collect(range(-5.2, 5.2, length = 320))
ap = plot(pbx, (pbx .^ 2) ./ 4; color = :gray, lw = 1.5, label = "parabola  Δ=τ²/4",
          xlabel = "τ", ylabel = "Δ", title = "train on a parabola window → how far it generalizes along the curve",
          legend = :top, xlims = (-5.6, 5.6), ylims = (-0.5, 7.2), titlefontsize = 11)
hline!(ap, [0.0]; color = :black, ls = :dot, label = "")
blue_l = RGBA(0.4, 0.6, 1.0, 0.9); red_l = RGBA(1.0, 0.55, 0.55, 0.9)
sel = [findfirst(==(F(v)), centers) for v in (-3.0, -0.75, 0.75, 3.0)]   # representative windows
for i in sel
    τc = Float64(centers[i]); s = sign(τc); dark = τc < 0 ? :blue : :red; light = τc < 0 ? blue_l : red_l
    oe = τc + s*Float64(w); oc = oe + s*r_outs[i]                        # outer edge, outward ε-crossing
    tx, ty = arc(oe, oc); plot!(ap, tx, ty; color = light, lw = 11, label = "")            # reach halo (under)
    tx, ty = arc(τc - Float64(w), τc + Float64(w)); plot!(ap, tx, ty; color = dark, lw = 11, label = "")  # train (over)
    nx, ny = pnrm(oc); plot!(ap, [oc-0.18nx, oc+0.18nx], [oc^2/4-0.18ny, oc^2/4+0.18ny]; color = dark, lw = 2, label = "")
    annotate!(ap, oc + s*0.15, oc^2/4 + 0.55, text("reach $(round(r_outs[i];sigdigits=2))", 8, dark, s < 0 ? :right : :left))
end
plot!(ap, [NaN], [NaN]; color = :blue, lw = 11, label = "training window (τ<0 stable)")
plot!(ap, [NaN], [NaN]; color = :red,  lw = 11, label = "training window (τ>0 unstable)")
plot!(ap, [NaN], [NaN]; color = RGBA(0.5, 0.5, 0.5, 0.9), lw = 11, label = "generalizes to here (rel-L2<ε) →")
savefig(ap, "data/parabola_slide_arcs.png")
println("\nPlot: data/parabola_slide_arcs.png")
