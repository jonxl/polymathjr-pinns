#=
TRAIN ON A SEGMENT OF THE PARABOLA  Delta = tau^2/4  (the discriminant-zero locus), then see
how error changes as you move ALONG the parabola away from the training segment.

The parabola tau^2 - 4 Delta = 0 is the repeated-root boundary between nodes (below, real
roots) and spirals (above, complex roots).  On it k = tau^2/4 - Delta = 0 exactly, and with
fixed ICs y(0)=1,y'(0)=0 the solution is the degenerate-node form
    u(x) = (1 - (tau/2) x) e^{(tau/2) x}.
So the parabola is a 1-PARAMETER family in tau: tau<0 = stable (decaying), tau>0 = unstable
(growing), vertex tau=0 = the origin.

Experiment: train on the segment tau in [-tau_train, tau_train] (points ON the curve), then
  [1] test ALONG the parabola for tau in [-Ltau, Ltau] -> rel-L2 vs tau (in- and out-of-segment).
      Training window is centered, so any left/right growth asymmetry is purely stable-vs-unstable.
  [2] a full-plane error map (region boundaries + parabola + training segment) -> transverse spread.
Unified-eig model; metric = solution-space relative L2.
Outputs: data/parabola_segment_curve.png , data/parabola_segment_map.png
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

tau_train = F(1.5)                 # train on parabola for tau in [-tau_train, tau_train]
n_train   = 300
Ltau      = 4.5                    # test along parabola for tau in [-Ltau, Ltau]
n_test    = 400
Lmap = 5.5; Ng = 81                # plane-map half-width (fits the parabola up to tau~4.7)
loss_floor = F(1e-6); patch_adam = 30000; patch_lbfgs = 60000

on_parab(t) = (t, t^2 / 4)         # a point on the parabola at abscissa tau=t

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
function pt_err(p, t, d)
    O = vec(first(net(reshape(F[t, d], 2, 1), p, st))); μ, k, A, B = O[1], O[2], O[3], O[4]
    up = [exp(μ*x)*(A*Cfun(k, x) + B*Sfun(k, x)) for x in xfine]; ut = utrue_vals(t, d, xfine)
    sqrt(sum(abs2, up .- ut) / (sum(abs2, ut) + 1e-12))
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
    return permutedims(reshape(errs, Ng, Ng))
end

# ===========================================================================
# Train on the central parabola segment
# ===========================================================================
tt = collect(range(-tau_train, tau_train, length = n_train))
ts = F.(tt); ds = F.((tt .^ 2) ./ 4)
Xtr = permutedims(hcat(ts, ds)); Utr = reduce(hcat, [utrue_vals(ts[k], ds[k], xs) for k in 1:n_train])
println("training on parabola segment tau in [-$tau_train, $tau_train] (n=$n_train), floor<$loss_floor")
p, fl = train_to_floor(Xtr, Utr, n_train)
println("final train loss = ", round(fl; sigdigits = 3), fl < F(1e-5) ? "  (converged)" : "  (UNDERTRAINED)")

# ---------------------------------------------------------------------------
# [1] error ALONG the parabola  vs tau
# ---------------------------------------------------------------------------
τtest = collect(range(-Ltau, Ltau, length = n_test))
errs = [pt_err(p, F(t), F(t^2 / 4)) for t in τtest]
inwin = abs.(τtest) .<= Float64(tau_train)
left  = τtest .< -Float64(tau_train)          # stable extrapolation wing
right = τtest .>  Float64(tau_train)          # unstable extrapolation wing
println("median rel-L2  in-segment = ", round(median(errs[inwin]); sigdigits = 3))
println("  stable wing (tau<0) median = ", round(median(errs[left]); sigdigits = 3),
        "   at tau=-$Ltau: ", round(errs[1]; sigdigits = 3))
println("  unstable wing (tau>0) median = ", round(median(errs[right]); sigdigits = 3),
        "   at tau=+$Ltau: ", round(errs[end]; sigdigits = 3))

pc = plot(τtest, max.(errs, 1e-6); yscale = :log10, lw = 2.5, color = :purple, legend = false,
          xlabel = "τ  along the parabola  (Δ = τ²/4)", ylabel = "solution rel-L2 (log)",
          title = "error along the parabola; train τ∈[-$tau_train,$tau_train]")
vspan!(pc, [-Float64(tau_train), Float64(tau_train)]; color = :green, alpha = 0.12, label = "")
hline!(pc, [eps_tol]; color = :red, ls = :dash, label = "")
vline!(pc, [0.0]; color = :gray, ls = :dot, label = "")     # vertex; left=stable, right=unstable
savefig(pc, "data/parabola_segment_curve.png")

# ---------------------------------------------------------------------------
# [2] full-plane error map: transverse spread off the parabola
# ---------------------------------------------------------------------------
Emap = plane_error_map(p)
hm = heatmap(τg, Δg, log10.(max.(Emap, 1e-4)); c = :viridis, colorbar_title = "log10 rel-L2",
             xlabel = "τ", ylabel = "Δ", title = "trained on parabola segment: error over the plane, ε=$eps_tol")
contour!(hm, τg, Δg, Emap; levels = [eps_tol], color = :red, lw = 2, colorbar_entry = false)
# region boundaries
plot!(hm, [-Lmap, Lmap], [0.0, 0.0]; color = :white, lw = 1, ls = :dash, label = "")
plot!(hm, [0.0, 0.0], [0.0, Lmap];   color = :white, lw = 1, ls = :dash, label = "")
pb = collect(range(-2*sqrt(Lmap), 2*sqrt(Lmap), length = 160))
plot!(hm, pb, (pb .^ 2) ./ 4; color = :white, lw = 1, ls = :dash, label = "parabola Δ=τ²/4")
# the training segment (bold) on the parabola
plot!(hm, Float64.(ts), Float64.(ds); color = :orange, lw = 4, label = "train segment")
savefig(hm, "data/parabola_segment_map.png")
println("\nPlots: data/parabola_segment_curve.png , data/parabola_segment_map.png")
