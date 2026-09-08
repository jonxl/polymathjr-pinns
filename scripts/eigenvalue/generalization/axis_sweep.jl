#=
AXIS SWEEP -- slide a single sunflower disk along an axis; a FRESH PINN is trained at each disk
position, and we record (a) its reach margin and (b) its full-plane error map.  Repeated for:
  * X-axis: disk centers (τ, 0) swept along the τ-axis
  * Y-axis: disk centers (0, Δ) swept along the Δ-axis
Original scale (Lmap=4), R_patch=0.5.  Unified-eig; solution-space rel-L2 (exact truth); robust.
Outputs:
  data/axis_sweep.png         reach margin vs position (X & Y)
  data/axis_sweep_maps_x.png  grid of per-disk error maps along the X-axis
  data/axis_sweep_maps_y.png  grid of per-disk error maps along the Y-axis
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

a0 = F(1.0); a1 = F(0.0)
pde_weight = F(1.0); ic_weight = F(1.0); sup_weight = F(1.0)
x_left = F(0.0); x_right = F(1.0); n_colloc = 50
xs = collect(range(x_left, x_right, length = n_colloc))
xfine = collect(range(x_left, x_right, length = 200)); Mf = length(xfine)
eps_tol = 0.10
R_patch = F(0.5); n_patch = 250
loss_floor = F(1e-6); patch_adam = 30000; patch_lbfgs = 60000
positions = collect(-3.0:0.75:3.0)                     # 9 disk positions per axis (3x3 map grid)
r_max = 6.0; r_step = 0.05; n_ang = 120                # reach march
Lm = 4.0; Ngm = 91                                     # error-map grid

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
utrue_vals(t, d, xpts) = (μ = t/2; k = t^2/4 - d; [usol(μ, k, 1.0, -μ, x) for x in Float64.(xpts)])
region(t, d) = d < 0 ? :saddle : (t^2-4d > 0 ? (t<0 ? :sn : :un) : (t<0 ? :ss : :us))
const GA = F(π * (3 - sqrt(5)))
function disk_at(cx, cy)
    ts = Vector{F}(undef, n_patch); ds = Vector{F}(undef, n_patch)
    for i in 1:n_patch
        r = R_patch * sqrt((F(i) - F(0.5)) / n_patch); θ = GA * i
        ts[i] = F(cx) + r * cos(θ); ds[i] = F(cy) + r * sin(θ)
    end
    return ts, ds
end
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
function pt_err(p, t, d)
    O = vec(first(net(reshape(F[t, d], 2, 1), p, st))); μ, k, A, B = Float64.(O)
    up = [usol(μ, k, A, B, x) for x in Float64.(xfine)]; ut = utrue_vals(Float64(t), Float64(d), xfine)
    e = sqrt(sum(abs2, up .- ut)/(sum(abs2, ut)+1e-12)); return isfinite(e) ? e : 1e3
end
function reach_margin(p, cx, cy)
    rs = Float64[]
    for θ in range(0, 2π, length = n_ang + 1)[1:end-1]
        r = 0.0
        while r < r_max; pt_err(p, F(cx + (r+r_step)*cos(θ)), F(cy + (r+r_step)*sin(θ))) > eps_tol && break; r += r_step; end
        push!(rs, r)
    end
    return median(rs) - Float64(R_patch)
end
τg = collect(range(-Lm, Lm, length = Ngm)); Δg = collect(range(-Lm, Lm, length = Ngm))
plane_map(p) = [pt_err(p, F(t), F(d)) for d in Δg, t in τg]
function overlay!(pl)
    plot!(pl, [-Lm, Lm], [0.0, 0.0]; color = :white, lw = 0.8, ls = :dash, label = "")
    plot!(pl, [0.0, 0.0], [-Lm, Lm]; color = :white, lw = 0.8, ls = :dash, label = "")
    tt = collect(range(-2*sqrt(Lm), 2*sqrt(Lm), length = 120)); plot!(pl, tt, (tt.^2)./4; color = :white, lw = 0.8, ls = :dash, label = "")
end
θc = range(0, 2π, length = 120)

# sweep both axes: fresh net per disk position, record reach + map
results = Dict{Symbol,Vector{Float64}}()
for (name, ax, fname) in [("X-axis (Δ=0, sweep τ)", :x, "x"), ("Y-axis (τ=0, sweep Δ)", :y, "y")]
    println(name); margins = Float64[]; panels = Plots.Plot[]
    for pos in positions
        cx, cy = ax == :x ? (pos, 0.0) : (0.0, pos)
        ts, ds = disk_at(cx, cy)
        Xtr = permutedims(hcat(ts, ds)); Utr = reduce(hcat, [F.(utrue_vals(Float64(ts[k]), Float64(ds[k]), xs)) for k in 1:n_patch])
        p = train_to_floor(Xtr, Utr, n_patch)
        m = reach_margin(p, cx, cy); push!(margins, m)
        println("  pos=", rpad(pos, 6), "reach margin = ", round(m; sigdigits = 3))
        Emap = plane_map(p)
        pl = heatmap(τg, Δg, log10.(max.(Emap, 1e-4)); c = :viridis, clims = (-4, 0.5), colorbar = false,
                     title = "center=($(round(cx;digits=2)),$(round(cy;digits=2)))  reach=$(round(m;digits=2))",
                     titlefontsize = 7, xlabel = "τ", ylabel = "Δ", aspect_ratio = :equal, xlims = (-Lm, Lm), ylims = (-Lm, Lm))
        contour!(pl, τg, Δg, Emap; levels = [eps_tol], color = :red, lw = 1.5, colorbar_entry = false)
        overlay!(pl); plot!(pl, cx .+ R_patch .* cos.(θc), cy .+ R_patch .* sin.(θc); color = :white, lw = 1.5, label = "")
        push!(panels, pl)
    end
    results[ax] = margins
    savefig(plot(panels...; layout = (3, 3), size = (1350, 1250), plot_title = name, plot_titlefontsize = 11),
            "data/axis_sweep_maps_$fname.png")
    println("  Plot: data/axis_sweep_maps_$fname.png")
end

pl = plot(xlabel = "disk-center position along axis", ylabel = "reach margin (median reach − R_patch)",
          title = "per-disk generalization vs position along each axis (ε=$eps_tol)", legend = :best)
plot!(pl, positions, results[:x]; lw = 2, marker = :circle, ms = 4, color = :dodgerblue, label = "X-axis (Δ=0): position = τ")
plot!(pl, positions, results[:y]; lw = 2, marker = :diamond, ms = 4, color = :firebrick, label = "Y-axis (τ=0): position = Δ")
vline!(pl, [0.0]; color = :gray, ls = :dot, label = "")
savefig(pl, "data/axis_sweep.png")
println("\nPlots: data/axis_sweep.png , data/axis_sweep_maps_x.png , data/axis_sweep_maps_y.png")
