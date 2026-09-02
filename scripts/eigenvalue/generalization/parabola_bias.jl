#=
PARABOLA BIAS -- quantify the left/right generalization bias vs where the training region sits.
For each training position τc along the parabola Δ=τ²/4, train a fresh PINN on the segment
[τc-w, τc+w], then measure the error at FIXED far test points τ = ±tau_test on the parabola:
    left_err  = rel-L2 at τ = -tau_test
    right_err = rel-L2 at τ = +tau_test
    bias      = right_err - left_err
Question: is |bias| proportional to the training region's distance from the origin (|τc|)?
Result: flat on the stable arm (~0.2); ~proportional to distance on the unstable arm (slope ~1.9).
Caveat: right_err pins at ~1 because the τ=+tau_test solution is enormous (~1e5), so the signal
is really the left error; single fixed points are noisier than a window average.

On the parabola k=τ²/4-Δ=0 (repeated root r=τ/2), so u=(1-(τ/2)x)e^{(τ/2)x}; exact stable truth.
Output: data/parabola_bias.png + table.
=#

using Lux
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote, ComponentArrays
using Plots
import Random
using Random: MersenneTwister
using Statistics: mean, median
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

w = 0.5; n_train = 120                                  # training segment half-width, points
loss_floor = F(1e-6); patch_adam = 30000; patch_lbfgs = 60000
taucs = collect(-3.5:0.5:3.5)                           # training-segment center positions along parabola
tau_test = 20.0                                        # FIXED test trace: left error at τ=-tau_test, right at +tau_test

const Pterm = 14
sxE = [F.(xs .^ (2n)) ./ F(factorial(big(2n))) for n in 0:Pterm]
sxO = [F.(xs .^ (2n + 1)) ./ F(factorial(big(2n + 1))) for n in 0:Pterm]
CS_xs(k) = (sum(sxE[n+1] .* (k .^ n) for n in 0:Pterm), sum(sxO[n+1] .* (k .^ n) for n in 0:Pterm))
function usol(μ, k, A, B, x)
    if k > 1e-12
        s = sqrt(k); return (A/2 + B/(2s))*exp((μ+s)*x) + (A/2 - B/(2s))*exp((μ-s)*x)
    elseif k < -1e-12
        ω = sqrt(-k); return exp(μ*x)*(A*cos(ω*x) + B*sin(ω*x)/ω)
    else
        return exp(μ*x)*(A + B*x)
    end
end
utrue_parab(t, xpts) = (μ = t/2; [usol(μ, 0.0, 1.0, -μ, x) for x in Float64.(xpts)])  # on parabola k=0

net = Lux.Chain(Lux.Dense(2, 64, tanh), Lux.Dense(64, 64, tanh), Lux.Dense(64, 64, tanh), Lux.Dense(64, 4))
_, st = Lux.setup(Random.default_rng(), net)
loss_core(p, Xtr, Utr, ntr) = begin
    O = first(net(Xtr, p, st)); μ = O[1:1, :]; k = O[2:2, :]; A = O[3:3, :]; B = O[4:4, :]
    τ = Xtr[1:1, :]; Δ = Xtr[2:2, :]
    C, S = CS_xs(k); v = A .* C .+ B .* S; vp = A .* (k .* S) .+ B .* C
    E = exp.(xs * μ); U = E .* v
    resid = E .* ((μ .^ 2 .+ k .- τ .* μ .+ Δ) .* v .+ (2 .* μ .- τ) .* vp)
    sum(abs2, resid)/(n_colloc*ntr) + (sum(abs2, A .- a0)+sum(abs2,(μ.*A.+B).-a1))/ntr + sum(abs2, U .- Utr)/(n_colloc*ntr)
end
function train_to_floor(Xtr, Utr, ntr)
    p0, _ = Lux.setup(MersenneTwister(1234), net); p0ca = ComponentArray(p0)
    prob = OptimizationProblem(OptimizationFunction((p, _) -> loss_core(p, Xtr, Utr, ntr), Optimization.AutoZygote()), p0ca)
    stopcb = (s, l) -> l < loss_floor
    r = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); callback = stopcb, maxiters = patch_adam)
    padam = r.u
    try
        rl = solve(remake(prob; u0 = padam), OptimizationOptimJL.LBFGS(); callback = stopcb, maxiters = patch_lbfgs)
        return isfinite(loss_core(rl.u, Xtr, Utr, ntr)) ? rl.u : padam
    catch; return padam; end
end
function err_parab(p, t)                                # rel-L2 at parabola point abscissa t
    d = t^2/4
    O = vec(first(net(reshape(F[t, d], 2, 1), p, st))); μ, k, A, B = Float64.(O)
    up = [usol(μ, k, A, B, x) for x in Float64.(xfine)]; ut = utrue_parab(t, xfine)
    e = sqrt(sum(abs2, up .- ut)/(sum(abs2, ut)+1e-12)); return isfinite(e) ? e : 1e3
end

lefts = Float64[]; rights = Float64[]; biases = Float64[]
println(rpad("τc", 8), rpad("err@τ=-$tau_test", 13), rpad("err@τ=+$tau_test", 13), rpad("bias(R-L)", 11), "|τc|")
for τc in taucs
    tt = collect(range(τc - w, τc + w, length = n_train))
    Xtr = permutedims(hcat(F.(tt), F.((tt .^ 2) ./ 4)))
    Utr = reduce(hcat, [F.(utrue_parab(tt[k], xs)) for k in 1:n_train])
    p = train_to_floor(Xtr, Utr, n_train)
    le = err_parab(p, F(-tau_test)); re = err_parab(p, F(tau_test))    # FIXED test points at τ = ∓tau_test
    push!(lefts, le); push!(rights, re); push!(biases, re - le)
    println(rpad(τc, 8), rpad(round(le; sigdigits=3), 13), rpad(round(re; sigdigits=3), 13),
            rpad(round(re-le; sigdigits=3), 11), round(abs(τc); sigdigits=3))
end

# fit bias vs |τc| (proportional test): bias ≈ a*|τc| + b
absc = abs.(taucs); A_ = hcat(absc, ones(length(absc))); coef = A_ \ biases
pred = A_ * coef; ss_res = sum((biases .- pred).^2); ss_tot = sum((biases .- mean(biases)).^2)
r2 = 1 - ss_res/ss_tot
println("\nfit bias ≈ $(round(coef[1];sigdigits=3))·|τc| + $(round(coef[2];sigdigits=3))   R²=$(round(r2;sigdigits=3))")
println("(proportional-to-distance would need intercept ≈ 0 and high R²)")

p1 = plot(taucs, lefts; lw=2, marker=:circle, ms=3, color=:blue, label="left error", xlabel="τc (training position)",
          ylabel="asymptotic rel-L2", title="error at fixed τ=±20 vs training position")
plot!(p1, taucs, rights; lw=2, marker=:circle, ms=3, color=:red, label="right error")
p2 = plot(taucs, biases; lw=2, marker=:diamond, ms=4, color=:purple, legend=false,
          xlabel="τc (training position)", ylabel="bias = right − left", title="bias vs training position")
hline!(p2, [0.0]; color=:gray, ls=:dash)
p3 = scatter(absc, biases; ms=5, color=:purple, label="data", xlabel="|τc|  (distance from origin)",
             ylabel="bias", title="bias vs distance  (R²=$(round(r2;digits=2)))")
plot!(p3, absc, pred; lw=2, color=:black, label="linear fit")
savefig(plot(p1, p2, p3; layout=(1,3), size=(1650,520), bottom_margin=8Plots.mm, left_margin=5Plots.mm), "data/parabola_bias.png")
println("\nPlot: data/parabola_bias.png")
