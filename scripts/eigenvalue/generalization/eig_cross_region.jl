#=
EIGENVALUE construction -- cross-region (in-family / out-of-family) transfer test.

Same idea as cross_region.jl (train one model per section, test on every section, build a
transfer matrix; diagonal = in-family, off-diagonal = out-of-family), but using the NEW
exponential representation from eig_family.jl:

    net : (tau, Delta) -> (lambda1, lambda2, c1, c2)
    u(x) = c1 e^{lambda1 x} + c2 e^{lambda2 x}

and the SOLUTION-space relative-L2 metric (basis-independent, the one we settled on).

Restricted to the REAL-eigenvalue (exponential) regions -- saddle / stable node /
unstable node -- because the two-real-exponential ansatz cannot represent the oscillatory
spiral/center solutions. So this is a 3x3 matrix.

The question: does this representation generalize across regions where the power-series
construction failed (off-diagonal 17-260%)? If the off-diagonal collapses here, that is
direct evidence the COEFFICIENT representation -- not the network -- blocked generalization.

We also report the pde / ic / sup component matrices, for parity with cross_region.jl.
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

regions = [:saddle, :stable_node, :unstable_node]

function region(tau, delta)
    delta < 0 && return :saddle
    delta == 0 && return :degenerate
    disc = tau^2 - 4 * delta
    tau == 0 && return :center
    disc > 0 && return tau < 0 ? :stable_node   : :unstable_node
    disc < 0 && return tau < 0 ? :stable_spiral : :unstable_spiral
    return :star
end

function utrue_vals(τ, Δ, xpts)
    s  = sqrt(Float64(τ)^2 - 4 * Float64(Δ))
    l1 = (Float64(τ) + s) / 2; l2 = (Float64(τ) - s) / 2
    d1 = (Float64(a1) - l2 * Float64(a0)) / (l1 - l2); d2 = Float64(a0) - d1
    return F[ d1 * exp(l1 * Float64(x)) + d2 * exp(l2 * Float64(x)) for x in xpts ]
end

function sample_region(reg, n, rng)
    ts = F[]; ds = F[]
    while length(ts) < n
        t = rand(rng, F) * (2 * tau_lim)   - tau_lim
        d = rand(rng, F) * (2 * delta_lim) - delta_lim
        region(t, d) == reg && (push!(ts, t); push!(ds, d))
    end
    return ts, ds
end
data_of(ts, ds) = (permutedims(hcat(ts, ds)),
                   reduce(hcat, [utrue_vals(ts[k], ds[k], xs) for k in 1:length(ts)]))

# ---------------------------------------------------------------------------
# Network + losses (exponential representation, same as eig_family.jl)
# ---------------------------------------------------------------------------
net = Lux.Chain(Lux.Dense(2, 64, tanh), Lux.Dense(64, 64, tanh),
                Lux.Dense(64, 64, tanh), Lux.Dense(64, 4))
_, st = Lux.setup(Random.default_rng(), net)

# reconstruct u at collocation points from the 4 params
function recon_U(p, Xb)
    O = first(net(Xb, p, st)); λ1 = O[1:1, :]; λ2 = O[2:2, :]; c1 = O[3:3, :]; c2 = O[4:4, :]
    return c1 .* exp.(xs * λ1) .+ c2 .* exp.(xs * λ2)
end

function pinn_loss(p, Xb, Ub, nb)
    O  = first(net(Xb, p, st)); λ1 = O[1:1, :]; λ2 = O[2:2, :]; c1 = O[3:3, :]; c2 = O[4:4, :]
    τ  = Xb[1:1, :]; Δ = Xb[2:2, :]
    E1 = exp.(xs * λ1); E2 = exp.(xs * λ2); U = c1 .* E1 .+ c2 .* E2
    resid = (c1 .* (λ1 .^ 2 .- τ .* λ1 .+ Δ)) .* E1 .+ (c2 .* (λ2 .^ 2 .- τ .* λ2 .+ Δ)) .* E2
    lp = sum(abs2, resid) / (n_colloc * nb)
    li = (sum(abs2, (c1 .+ c2) .- a0) + sum(abs2, (λ1 .* c1 .+ λ2 .* c2) .- a1)) / nb
    ls = sup_weight == 0 ? zero(F) : sum(abs2, U .- Ub) / (n_colloc * nb)
    return pde_weight * lp + ic_weight * li + sup_weight * ls
end

function loss_components(p, Xb, Ub, nb)
    O  = first(net(Xb, p, st)); λ1 = O[1:1, :]; λ2 = O[2:2, :]; c1 = O[3:3, :]; c2 = O[4:4, :]
    τ  = Xb[1:1, :]; Δ = Xb[2:2, :]
    E1 = exp.(xs * λ1); E2 = exp.(xs * λ2); U = c1 .* E1 .+ c2 .* E2
    resid = (c1 .* (λ1 .^ 2 .- τ .* λ1 .+ Δ)) .* E1 .+ (c2 .* (λ2 .^ 2 .- τ .* λ2 .+ Δ)) .* E2
    lp = sum(abs2, resid) / (n_colloc * nb)
    li = (sum(abs2, (c1 .+ c2) .- a0) + sum(abs2, (λ1 .* c1 .+ λ2 .* c2) .- a1)) / nb
    ls = sum(abs2, U .- Ub) / (n_colloc * nb)
    return lp, li, ls
end

# mean per-ODE SOLUTION-space relative L2
function solspace_relL2(p, Xb, Ub)
    Up = recon_U(p, Xb)
    mean(sqrt.(sum(abs2, Up .- Ub; dims = 1) ./ (sum(abs2, Ub; dims = 1) .+ F(1e-8))))
end

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
        println(rpad(string(r), 18),
                join([rpad(round(Mat[i, j]; sigdigits = 3), 16) for j in 1:nr]))
    end
    hm = heatmap(labels, labels, log10.(max.(Mat, 1e-20));
                 yflip = true, c = :viridis, colorbar_title = "log10 value",
                 xlabel = "tested on", ylabel = "trained on", xrotation = 20,
                 title = ttl, size = (640, 520))
    for i in 1:nr, j in 1:nr
        txt = pct ? string(round(Mat[i, j] * 100; sigdigits = 2), "%") :
                    string(round(Mat[i, j]; sigdigits = 2))
        annotate!(hm, j, i, text(txt, 8, :white))
    end
    savefig(hm, fname)
    println("Plot saved: $fname")
end

report_matrix(M,    "data/eig_cross_region_matrix.png", "eig transfer: solution-space rel-L2 (diag = in-family)"; pct = true)
report_matrix(Mpde, "data/eig_cross_region_pde.png",    "eig: PDE residual component (raw)")
report_matrix(Mic,  "data/eig_cross_region_ic.png",     "eig: IC component (raw)")
report_matrix(Msup, "data/eig_cross_region_sup.png",    "eig: supervised component (raw)")
