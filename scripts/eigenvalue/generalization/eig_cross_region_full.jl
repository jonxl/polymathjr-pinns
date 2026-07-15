#=
UNIFIED (all-6-region) cross-region transfer test.

The two-real-exponential ansatz (eig_cross_region.jl) cannot represent the oscillatory
spiral/center solutions, so it was limited to the 3 real-root regions. This script uses a
UNIFIED form that covers real AND complex roots smoothly.

Substituting u = e^{μ x} v with μ = τ/2 turns y'' - τ y' + Δ y = 0 into v'' = k v, where
k = τ^2/4 - Δ (the discriminant / 4). Then

    u(x) = e^{μ x} [ A * C(k,x) + B * S(k,x) ],   C(k,x)=cosh(√k x),  S(k,x)=sinh(√k x)/√k

  * k > 0 (real roots)    -> cosh/sinh  (exponential, saddle/nodes)
  * k < 0 (complex roots) -> cos/sin    (oscillatory, spirals/center)
  * C,S are ENTIRE functions of k, so this is smooth across the discriminant (k=0).

C,S are evaluated by their (fast-converging, branch-free) power series in k, so the whole
thing is real-valued and differentiable for any sign of k.

Network: (tau, Delta) -> (mu, k, A, B).
Loss (analytic, well conditioned -- no high-order differentiation matrices):
  loss_pde : residual = e^{μx}[(μ²+k-τμ+Δ) v + (2μ-τ) v'],  v=A C+B S, v'=A k S+B C
             -> drives μ->τ/2, k->τ²/4-Δ.
  loss_ic  : u(0)=A -> a0,  u'(0)=μA+B -> a1
  loss_sup : SOLUTION-space MSE vs analytic true u.
Metric: solution-space relative L2. All 6 trace-det regions.
=#

using Lux
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote, ComponentArrays
using Plots
import Random
using Random: MersenneTwister
using Statistics: mean
using LinearAlgebra

isdir("data") || mkpath("data")
F = Float32

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
a0 = F(1.0); a1 = F(0.0)
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)
tau_lim = F(2.0); delta_lim = F(2.0)
n_train = 600; n_test = 300
x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))
adam_iters  = 3000
lbfgs_iters = 2000

regions = [:saddle, :stable_node, :unstable_node, :stable_spiral, :unstable_spiral, :center]

function region(tau, delta)
    delta < 0 && return :saddle
    delta == 0 && return :degenerate
    disc = tau^2 - 4 * delta
    tau == 0 && return :center
    disc > 0 && return tau < 0 ? :stable_node   : :unstable_node
    disc < 0 && return tau < 0 ? :stable_spiral : :unstable_spiral
    return :star
end

# ---------------------------------------------------------------------------
# C(k,x)=cosh(√k x), S(k,x)=sinh(√k x)/√k via power series in k (works for k<0 -> cos/sin).
# Pterm=14 (=> up to x^29) is exact to machine precision for |k| x^2 <~ 6, our range.
# ---------------------------------------------------------------------------
const Pterm = 14
cC = F.([1 / factorial(big(2n))     for n in 0:Pterm])     # 1/(2n)!
cS = F.([1 / factorial(big(2n + 1)) for n in 0:Pterm])     # 1/(2n+1)!
xpowE = [F.(xs .^ (2n))     for n in 0:Pterm]              # x^{2n}   (n_colloc vectors)
xpowO = [F.(xs .^ (2n + 1)) for n in 0:Pterm]              # x^{2n+1}

# C,S as n_colloc × nb, from k (1×nb)
CS_series(k) = (sum(cC[n+1] .* (k .^ n) .* xpowE[n+1] for n in 0:Pterm),
                sum(cS[n+1] .* (k .^ n) .* xpowO[n+1] for n in 0:Pterm))

# scalar closed forms for the ANALYTIC true solution (exact, no AD needed)
Cfun(k, x) = k >= 0 ? cosh(sqrt(k) * x) : cos(sqrt(-k) * x)
Sfun(k, x) = abs(k) < 1e-12 ? x : (k > 0 ? sinh(sqrt(k) * x) / sqrt(k) : sin(sqrt(-k) * x) / sqrt(-k))
function utrue_vals(τ, Δ, xpts)
    μ = Float64(τ) / 2; k = Float64(τ)^2 / 4 - Float64(Δ)
    A = Float64(a0); B = Float64(a1) - μ * Float64(a0)          # u(0)=A, u'(0)=μA+B
    return F[ exp(μ * Float64(x)) * (A * Cfun(k, Float64(x)) + B * Sfun(k, Float64(x))) for x in xpts ]
end

# ---------------------------------------------------------------------------
# Region sampling (center is the tau=0 axis)
# ---------------------------------------------------------------------------
function sample_region(reg, n, rng)
    ts = F[]; ds = F[]
    if reg == :center
        while length(ts) < n
            push!(ts, F(0.0)); push!(ds, rand(rng, F) * delta_lim)
        end
    else
        while length(ts) < n
            t = rand(rng, F) * (2 * tau_lim)   - tau_lim
            d = rand(rng, F) * (2 * delta_lim) - delta_lim
            region(t, d) == reg && (push!(ts, t); push!(ds, d))
        end
    end
    return ts, ds
end
data_of(ts, ds) = (permutedims(hcat(ts, ds)),
                   reduce(hcat, [utrue_vals(ts[k], ds[k], xs) for k in 1:length(ts)]))

# ---------------------------------------------------------------------------
# Network + losses (unified representation)
# ---------------------------------------------------------------------------
net = Lux.Chain(Lux.Dense(2, 64, tanh), Lux.Dense(64, 64, tanh),
                Lux.Dense(64, 64, tanh), Lux.Dense(64, 4))
_, st = Lux.setup(Random.default_rng(), net)

function recon_U(p, Xb)
    O = first(net(Xb, p, st)); μ = O[1:1, :]; k = O[2:2, :]; A = O[3:3, :]; B = O[4:4, :]
    C, S = CS_series(k)
    return exp.(xs * μ) .* (A .* C .+ B .* S)
end

function pinn_loss(p, Xb, Ub, nb)
    O = first(net(Xb, p, st)); μ = O[1:1, :]; k = O[2:2, :]; A = O[3:3, :]; B = O[4:4, :]
    τ = Xb[1:1, :]; Δ = Xb[2:2, :]
    C, S = CS_series(k)
    v  = A .* C .+ B .* S
    vp = A .* (k .* S) .+ B .* C                 # v' = A k S + B C
    E  = exp.(xs * μ)
    U  = E .* v
    resid = E .* ((μ .^ 2 .+ k .- τ .* μ .+ Δ) .* v .+ (2 .* μ .- τ) .* vp)
    lp = sum(abs2, resid) / (n_colloc * nb)
    li = (sum(abs2, A .- a0) + sum(abs2, (μ .* A .+ B) .- a1)) / nb
    ls = sup_weight == 0 ? zero(F) : sum(abs2, U .- Ub) / (n_colloc * nb)
    return pde_weight * lp + ic_weight * li + sup_weight * ls
end

function loss_components(p, Xb, Ub, nb)
    O = first(net(Xb, p, st)); μ = O[1:1, :]; k = O[2:2, :]; A = O[3:3, :]; B = O[4:4, :]
    τ = Xb[1:1, :]; Δ = Xb[2:2, :]
    C, S = CS_series(k); v = A .* C .+ B .* S; vp = A .* (k .* S) .+ B .* C
    E = exp.(xs * μ); U = E .* v
    resid = E .* ((μ .^ 2 .+ k .- τ .* μ .+ Δ) .* v .+ (2 .* μ .- τ) .* vp)
    lp = sum(abs2, resid) / (n_colloc * nb)
    li = (sum(abs2, A .- a0) + sum(abs2, (μ .* A .+ B) .- a1)) / nb
    ls = sum(abs2, U .- Ub) / (n_colloc * nb)
    return lp, li, ls
end

solspace_relL2(p, Xb, Ub) =
    mean(sqrt.(sum(abs2, recon_U(p, Xb) .- Ub; dims = 1) ./ (sum(abs2, Ub; dims = 1) .+ F(1e-8))))

# ---------------------------------------------------------------------------
# Train one model per region
# ---------------------------------------------------------------------------
trained = Dict{Symbol,Any}()
for reg in regions
    rng = MersenneTwister(1234)
    ts, ds = sample_region(reg, n_train, rng)
    Xtr, Utr = data_of(ts, ds)
    p0, _ = Lux.setup(rng, net); p0ca = ComponentArray(p0)
    lf(p, _) = pinn_loss(p, Xtr, Utr, n_train)
    prob = OptimizationProblem(OptimizationFunction(lf, Optimization.AutoZygote()), p0ca)
    r = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); maxiters = adam_iters)
    r = solve(remake(prob; u0 = r.u), OptimizationOptimJL.LBFGS(); maxiters = lbfgs_iters)
    trained[reg] = r.u
    println("trained on $reg")
end

# ---------------------------------------------------------------------------
# Transfer matrices: train (row) x test (col)
# ---------------------------------------------------------------------------
rng = MersenneTwister(99)
test_sets = Dict(reg => data_of(sample_region(reg, n_test, rng)...) for reg in regions)

nr = length(regions)
M = zeros(nr, nr); Mpde = zeros(nr, nr); Mic = zeros(nr, nr); Msup = zeros(nr, nr)
for (i, tr) in enumerate(regions), (j, te) in enumerate(regions)
    Xte, Ute = test_sets[te]
    M[i, j] = solspace_relL2(trained[tr], Xte, Ute)
    lp, li, ls = loss_components(trained[tr], Xte, Ute, n_test)
    Mpde[i, j] = lp; Mic[i, j] = li; Msup[i, j] = ls
end

labels = string.(regions)
function report_matrix(Mat, fname, ttl; pct = false)
    println("\nrows = trained on, cols = tested on  ($ttl)")
    println(rpad("", 18), join([rpad(string(r), 16) for r in regions]))
    for (i, r) in enumerate(regions)
        println(rpad(string(r), 18), join([rpad(round(Mat[i, j]; sigdigits = 3), 16) for j in 1:nr]))
    end
    hm = heatmap(labels, labels, log10.(max.(Mat, 1e-20));
                 yflip = true, c = :viridis, colorbar_title = "log10 value",
                 xlabel = "tested on", ylabel = "trained on", xrotation = 30,
                 title = ttl, size = (820, 680))
    for i in 1:nr, j in 1:nr
        txt = pct ? string(round(Mat[i, j] * 100; sigdigits = 2), "%") : string(round(Mat[i, j]; sigdigits = 2))
        annotate!(hm, j, i, text(txt, 7, :white))
    end
    savefig(hm, fname)
    println("Plot saved: $fname")
end

report_matrix(M,    "data/eig_full_matrix.png", "unified eig transfer: solution rel-L2 (diag = in-family)"; pct = true)
report_matrix(Mpde, "data/eig_full_pde.png",    "unified eig: PDE residual component (raw)")
report_matrix(Mic,  "data/eig_full_ic.png",     "unified eig: IC component (raw)")
report_matrix(Msup, "data/eig_full_sup.png",    "unified eig: supervised component (raw)")

# ---------------------------------------------------------------------------
# Grouped-bar summary (non-heatmap): in-family (diagonal) vs out-of-family
# (off-diagonal), GEOMETRIC mean per component. For the eig representation the
# residual has no u'' amplification, so expect the three blow-ups to be SIMILAR
# (unlike the power-series version, where PDE dominates).
# ---------------------------------------------------------------------------
geomean(v)        = exp(mean(log.(max.(v, 1e-20))))
diag_vals(Mat)    = [Mat[i, i] for i in 1:nr]
offdiag_vals(Mat) = [Mat[i, j] for i in 1:nr for j in 1:nr if i != j]
comp_mats  = [Mpde, Mic, Msup]; comp_names = ["PDE residual", "IC", "supervised"]
in_fam  = [geomean(diag_vals(Mat))    for Mat in comp_mats]
out_fam = [geomean(offdiag_vals(Mat)) for Mat in comp_mats]
println("\ncomponent      in-family      out-of-family    blow-up")
for k in 1:3
    println(rpad(comp_names[k], 15), rpad(round(in_fam[k]; sigdigits = 3), 15),
            rpad(round(out_fam[k]; sigdigits = 3), 17), "×", round(out_fam[k] / in_fam[k]; sigdigits = 2))
end
xp = collect(1:3)
bpc = bar(xp .- 0.2, in_fam; bar_width = 0.4, label = "in-family (diagonal)", color = :seagreen,
          yscale = :log10, xticks = (xp, comp_names), legend = :topright,
          ylabel = "loss component (raw, geometric mean)",
          title = "eig cross-region components: in-family vs out-of-family", size = (860, 560))
bar!(bpc, xp .+ 0.2, out_fam; bar_width = 0.4, label = "out-of-family (off-diagonal)", color = :firebrick)
for k in 1:3
    annotate!(bpc, k, out_fam[k] * 4, text("×$(round(out_fam[k]/in_fam[k]; sigdigits = 2))", 8))
end
savefig(bpc, "data/eig_full_component_bars.png")
println("Plot saved: data/eig_full_component_bars.png")

# ---------------------------------------------------------------------------
# Combined comparison: power-series vs eig, out-of-family component geomeans.
# Power-series values are precomputed from the cross_region.jl run.
# NOTE: only the PDE term is a like-for-like ODE-violation measure across the two;
# IC/supervised measure different quantities (coeff-space vs solution-space), so the
# robust read is the WITHIN-representation ordering -- PDE flips from #1 (power series,
# it dominates) to #3 (eig, it is the smallest).
# ---------------------------------------------------------------------------
ps_out  = [13.7, 0.144, 0.0689]     # power-series out-of-family geomean: PDE, IC, sup
eig_out = out_fam                    # eig out-of-family geomean (computed above)
xp2 = collect(1:3)
bpcmp = bar(xp2 .- 0.2, ps_out; bar_width = 0.4, label = "power series", color = :orange,
            yscale = :log10, xticks = (xp2, comp_names), legend = :topright,
            ylabel = "out-of-family component (raw geomean, log)",
            title = "Out-of-family loss components: power-series vs eig", size = (860, 560))
bar!(bpcmp, xp2 .+ 0.2, eig_out; bar_width = 0.4, label = "eig", color = :dodgerblue)
savefig(bpcmp, "data/component_bars_compare.png")
println("Plot saved: data/component_bars_compare.png")
