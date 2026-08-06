#=
GENERALIZING PINN over the trace-determinant plane.

Same idea as generalize_tracedet.jl (network conditioned on the ODE via (tau,Delta)),
but trained like the ORIGINAL PINN: the loss is the ODE RESIDUAL plus initial-condition
penalties -- NOT supervised labels. So in pure-physics mode the network is never told
the coefficients; it must output a series that SATISFIES the equation.

ODE family (2nd-order linear, fixed ICs):
        y'' - tau*y' + Delta*y = 0,    y(0)=1, y'(0)=0

Network:  (tau, Delta) -> [c_0, ..., c_N]   (series coefficients, u(x)=sum c_n x^n)

Three weighted loss terms (knobs, exactly like the original PINN):
    loss = pde_weight * loss_pde            # mean residual^2 over collocation points
         + ic_weight  * loss_ic            # (c_0-1)^2 + (c_1-0)^2
         + sup_weight * loss_sup           # MSE vs true c_n  (the only term using labels)
Set sup_weight=0 for a pure PINN; turn it up to blend in supervision.

The true coefficients (from the recurrence a_{n+2}=tau a_{n+1}-Delta a_n) are computed
ONLY to score the result -- they never enter the loss unless sup_weight>0.
=#

using Lux
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote
using ComponentArrays
using Plots, ProgressMeter
import Random
using LinearAlgebra

isdir("data") || mkpath("data")

F = Float32
Random.seed!(1234)

# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------
N  = 10                      # series degree: c_0 .. c_N
a0 = F(1.0)                  # initial conditions  y(0)=a0, y'(0)=a1
a1 = F(0.0)

# --- the three loss-term knobs ---
pde_weight = F(1.0)          # ODE residual term
ic_weight  = F(1.0)          # initial-condition term
sup_weight = F(1.0)          # supervised term (0 = pure PINN; >0 blends in labels)

# (tau, Delta) sampling box and region restriction (same as before).
tau_lim   = F(2.0)
delta_lim = F(2.0)
n_total   = 600

split_mode = :interpolation          # :interpolation or :extrapolation
in_spiral(tau, delta) = (delta > 0) && (tau^2 - 4*delta < 0)

# Restrict to ONE trace-determinant section (or :all for the whole plane).
# :saddle, :stable_node, :unstable_node, :stable_spiral, :unstable_spiral, :all
restrict_to_region = :stable_spiral

# Collocation points where the residual is enforced (the "physics" domain).
x_left   = F(0.0)
x_right  = F(1.0)
n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))

# ---------------------------------------------------------------------------
# True coefficients (for SCORING; also used by sup term if sup_weight>0).
# Recurrence gives derivative values a_n; we store series coeffs c_n = a_n/n!.
# ---------------------------------------------------------------------------
fact_vec = F.(factorial.(0:N))
function true_cn(tau, delta)
    a = zeros(F, N + 1)
    a[1] = a0; a[2] = a1
    for n in 0:(N - 2)
        a[n + 3] = tau * a[n + 2] - delta * a[n + 1]
    end
    return a ./ fact_vec            # c_n = a_n / n!
end

function region(tau, delta)
    delta < 0 && return :saddle
    delta == 0 && return :degenerate
    disc = tau^2 - 4 * delta
    tau == 0 && return :center
    disc > 0 && return tau < 0 ? :stable_node   : :unstable_node
    disc < 0 && return tau < 0 ? :stable_spiral : :unstable_spiral
    return :star
end
in_region(tau, delta) = restrict_to_region == :all || region(tau, delta) == restrict_to_region

# ---------------------------------------------------------------------------
# Dataset: sample ODEs (rejection-sample into the chosen region)
# ---------------------------------------------------------------------------
taus = F[]; deltas = F[]
while length(taus) < n_total
    t = rand(F) * (2 * tau_lim)   - tau_lim
    d = rand(F) * (2 * delta_lim) - delta_lim
    in_region(t, d) && (push!(taus, t); push!(deltas, d))
end
println("region=$restrict_to_region   sampled $n_total ODEs")

X = permutedims(hcat(taus, deltas))                                   # 2 × n_total
Y = reduce(hcat, [true_cn(taus[k], deltas[k]) for k in 1:n_total])    # (N+1) × n_total  (true c_n)

if split_mode == :interpolation
    perm = Random.shuffle(1:n_total); n_test = round(Int, 0.2 * n_total)
    test_idx = perm[1:n_test]; train_idx = perm[n_test+1:end]
elseif split_mode == :extrapolation
    test_idx  = [k for k in 1:n_total if in_spiral(taus[k], deltas[k])]
    train_idx = [k for k in 1:n_total if !in_spiral(taus[k], deltas[k])]
else
    error("unknown split_mode")
end
X_tr, Y_tr = X[:, train_idx], Y[:, train_idx]
X_te, Y_te = X[:, test_idx],  Y[:, test_idx]
n_tr, n_te = length(train_idx), length(test_idx)
println("split=$split_mode   train=$n_tr   test=$n_te")

# ---------------------------------------------------------------------------
# Power matrices for evaluating u, u', u'' from the coefficients at collocation
# points (constants, built once). For c_n with u(x)=sum c_n x^n:
#   u   term i: x^(i-1)
#   u'  term i: (i-1) x^(i-2)
#   u'' term i: (i-1)(i-2) x^(i-3)
# (guard the negative-power entries to 0)
# ---------------------------------------------------------------------------
Pu  = F[ xs[m]^(i - 1)                                  for m in 1:n_colloc, i in 1:N+1 ]
Pu1 = F[ (i - 1) >= 1 ? (i - 1) * xs[m]^(i - 2) : 0     for m in 1:n_colloc, i in 1:N+1 ]
Pu2 = F[ (i - 1) >= 2 ? (i - 1)*(i - 2)*xs[m]^(i - 3) : 0 for m in 1:n_colloc, i in 1:N+1 ]

# ---------------------------------------------------------------------------
# Network: (tau, Delta) -> c_n
# ---------------------------------------------------------------------------
net = Lux.Chain(
    Lux.Dense(2,  64, tanh),
    Lux.Dense(64, 64, tanh),
    Lux.Dense(64, 64, tanh),
    Lux.Dense(64, N + 1)
)
p_init, st = Lux.setup(Random.default_rng(), net)
p_init_ca  = ComponentArray(p_init)

# ---------------------------------------------------------------------------
# PINN loss over a batch of ODEs (columns of Xb; their true c_n in Yb)
# ---------------------------------------------------------------------------
function pinn_loss(p, Xb, Yb, nb)
    C  = first(net(Xb, p, st))            # (N+1) × nb   predicted c_n
    τ  = Xb[1:1, :]                        # 1 × nb
    Δ  = Xb[2:2, :]                        # 1 × nb

    # Series and derivatives at all collocation points, for all ODEs at once.
    U  = Pu  * C                           # n_colloc × nb   u
    U1 = Pu1 * C                           #                 u'
    U2 = Pu2 * C                           #                 u''

    resid = U2 .- τ .* U1 .+ Δ .* U        # n_colloc × nb   should be ~0
    loss_pde = sum(abs2, resid) / (n_colloc * nb)

    loss_ic  = (sum(abs2, C[1, :] .- a0) + sum(abs2, C[2, :] .- a1)) / nb

    loss_sup = sup_weight == 0 ? zero(F) :
               sum(abs2, C .- Yb) / ((N + 1) * nb)

    return pde_weight * loss_pde + ic_weight * loss_ic + sup_weight * loss_sup
end
loss_fn(p, _) = pinn_loss(p, X_tr, Y_tr, n_tr)

# Same three terms, returned SEPARATELY (raw / unweighted) for logging & plotting.
function loss_components(p, Xb, Yb, nb)
    C  = first(net(Xb, p, st))
    τ  = Xb[1:1, :]; Δ = Xb[2:2, :]
    resid    = (Pu2 * C) .- τ .* (Pu1 * C) .+ Δ .* (Pu * C)
    loss_pde = sum(abs2, resid) / (n_colloc * nb)
    loss_ic  = (sum(abs2, C[1, :] .- a0) + sum(abs2, C[2, :] .- a1)) / nb
    loss_sup = sum(abs2, C .- Yb) / ((N + 1) * nb)
    return loss_pde, loss_ic, loss_sup
end

# RMS relative coefficient error vs the true c_n (for honest reporting).
function rel_rmse(p, Xb, Yb, nb)
    C = first(net(Xb, p, st))
    diff2 = sum(abs2, C .- Yb; dims = 1)
    nrm2  = sum(abs2, Yb; dims = 1) .+ F(1e-8)
    return sqrt(sum(diff2 ./ nrm2) / nb)
end

# ---------------------------------------------------------------------------
# Train (Adam -> LBFGS); log RMS relative error on train & test each step.
# ---------------------------------------------------------------------------
maxiters = 2000
p_bar = Progress(maxiters, desc = "PINN over (tau,Delta) ...")
iter_count = 0
train_hist = Float64[]; test_hist = Float64[]
pde_hist = Float64[]; ic_hist = Float64[]; sup_hist = Float64[]   # loss components
shown = [(:iter, 0), (:train_relerr, 0.0), (:test_relerr, 0.0)]
function log_components(p)
    lp, li, ls = loss_components(p, X_tr, Y_tr, n_tr)
    push!(pde_hist, lp); push!(ic_hist, li); push!(sup_hist, ls)
end
callback = function (state, l)
    global iter_count += 1; global shown
    push!(train_hist, rel_rmse(state.u, X_tr, Y_tr, n_tr))
    push!(test_hist,  rel_rmse(state.u, X_te, Y_te, n_te))
    log_components(state.u)
    if iter_count % 200 == 0 || iter_count == 1 || iter_count == maxiters
        shown = [(:iter, iter_count), (:loss, l),
                 (:train_relerr, train_hist[end]), (:test_relerr, test_hist[end])]
    end
    ProgressMeter.next!(p_bar; showvalues = shown)
    return false
end

adtype = Optimization.AutoZygote()
optfun = OptimizationFunction(loss_fn, adtype)
prob   = OptimizationProblem(optfun, p_init_ca)

res = solve(prob, OptimizationOptimisers.Adam(F(1e-3)); callback = callback, maxiters = maxiters)

n_adam = length(train_hist)
maxiters_lbfgs = 800
p_bar2 = Progress(maxiters_lbfgs, desc = "LBFGS fine-tune ...")
iter2 = 0
cb2 = function (state, l)
    global iter2 += 1
    push!(train_hist, rel_rmse(state.u, X_tr, Y_tr, n_tr))
    push!(test_hist,  rel_rmse(state.u, X_te, Y_te, n_te))
    log_components(state.u)
    ProgressMeter.next!(p_bar2; showvalues = [(:iter, iter2), (:loss, l),
                        (:train_relerr, train_hist[end]), (:test_relerr, test_hist[end])])
    return false
end
prob2 = remake(prob; u0 = res.u)
res   = solve(prob2, OptimizationOptimJL.LBFGS(); callback = cb2, maxiters = maxiters_lbfgs)
p_trained = res.u

# ---------------------------------------------------------------------------
# Evaluate & plot
# ---------------------------------------------------------------------------
pred_all = first(net(X, p_trained, st))
rel_err  = vec(sqrt.(sum(abs2, pred_all .- Y; dims = 1) ./ (sum(abs2, Y; dims = 1) .+ F(1e-8))))
is_test  = falses(n_total); is_test[test_idx] .= true

println("\nweights: pde=$pde_weight ic=$ic_weight sup=$sup_weight")
println("Final train rel-rmse = ", round(train_hist[end]; sigdigits = 4))
println("Final test  rel-rmse = ", round(test_hist[end];  sigdigits = 4))

# --- relative error vs iteration ---
plot_loss = plot(1:length(train_hist), max.(train_hist, 1e-20), yscale = :log10,
                 label = "train", lw = 2, title = "RMS rel error vs Iteration (PINN)",
                 xlabel = "Iteration (Adam then LBFGS)", ylabel = "rel-RMSE")
plot!(plot_loss, 1:length(test_hist), max.(test_hist, 1e-20), label = "test", lw = 2)
vline!(plot_loss, [n_adam + 0.5], ls = :dash, color = :gray, label = "Adam | LBFGS")
savefig(plot_loss, "data/generalize_pinn_loss.png")

# --- the three loss components vs iteration (raw, unweighted) ---
its = 1:length(pde_hist)
plot_comp = plot(its, max.(pde_hist, 1e-20), yscale = :log10, lw = 2, label = "pde residual",
                 title = "Loss components vs iteration (raw; weights pde=$pde_weight ic=$ic_weight sup=$sup_weight)",
                 xlabel = "Iteration (Adam then LBFGS)", ylabel = "component value")
plot!(plot_comp, its, max.(ic_hist, 1e-20), lw = 2, label = "ic / bc")
plot!(plot_comp, its, max.(sup_hist, 1e-20), lw = 2, label = "supervised")
vline!(plot_comp, [n_adam + 0.5], ls = :dash, color = :gray, label = "Adam | LBFGS")
savefig(plot_comp, "data/generalize_pinn_components.png")

# --- error across the plane ---
logerr = log10.(max.(rel_err, F(1e-12)))
err_plot = scatter(taus[.!is_test], deltas[.!is_test]; marker_z = logerr[.!is_test],
                   markershape = :circle, ms = 4, label = "train",
                   colorbar_title = "log10 rel error", xlabel = "τ (trace)",
                   ylabel = "Δ (determinant)",
                   title = "PINN generalization ($split_mode, region=$restrict_to_region)")
scatter!(err_plot, taus[is_test], deltas[is_test]; marker_z = logerr[is_test],
         markershape = :diamond, ms = 6, markerstrokewidth = 1.5, label = "test")
τg = range(-tau_lim, tau_lim, length = 200)
plot!(err_plot, τg, (τg .^ 2) ./ 4, color = :black, lw = 2, label = "τ²=4Δ")
hline!(err_plot, [0.0], color = :black, ls = :dash, lw = 1, label = "Δ=0")
savefig(err_plot, "data/generalize_pinn_error_map.png")

println("\nPlots: data/generalize_pinn_loss.png , data/generalize_pinn_components.png , data/generalize_pinn_error_map.png")
