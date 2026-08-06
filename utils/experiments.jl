# ===================================================================
# Experiments — the three orchestration shapes.
#
# Every chart in this project (yours and jeet's, both representations)
# is produced by one of three patterns. Encoding them once means the
# experiment scripts become thin configuration rather than fifteen
# copies of a training loop:
#
#   Shape A  run_single    train 1 model; chart its history + final state
#   Shape B  run_sweep     train N models varying ONE parameter;
#                          chart metric vs that parameter
#   Shape C  run_transfer  train N models (one per training set), evaluate
#                          each against M test sets; chart the N×M matrix
#   Shape D  run_extrapolate  train per-region in-box, test on concentric
#                             shells of the same region; error-vs-R curves
#   Shape E  run_cross_region_range  train per-region, test on every region
#                                    at shells; family + range combined
#   Shape F  run_showcase   train on exponential family, test 6 scenarios;
#                           solution curves + rel-L2 bar per scenario
#   Shape G  run_gen_radius  train on disk/family, evaluate on grid plane;
#                            error-map + ε-contour + generalization area
#
# Each takes `representation` (:power_series | :eigenvalue), so every
# experiment runs in both and the viewer's dropdown compares them.
#
# Each returns a PanelSet — never a PNG. Rendering is the viewer's job
# (see viz/Panels.jl); this module only produces data.
# ===================================================================

module Experiments

using Random
using Statistics: mean

include("../architectures/PINN.jl")
using .PINN

include("plugboard.jl")
using .Plugboard

include("../viz/PanelSpec.jl")
using .PanelSpec

const LF = PINN.loss_functions

export run_single, run_sweep, run_transfer, evaluate_on, ExperimentConfig,
       run_extrapolate, run_cross_region_range, run_showcase, run_gen_radius,
       save_experiment, load_experiment

"""
    save_experiment(path, panelset)

Write a PanelSet to JSON so it can be loaded by the viewer.
"""
save_experiment(path, ps) = PanelSpec.save_panels(path, ps)

"""
    load_experiment(path) → PanelSet

Read a PanelSet from JSON.
"""
load_experiment(path) = PanelSpec.load_panels(path)

"""
    ExperimentConfig

Everything shared by the three shapes. Kept separate from `PINNSettings`
because a single experiment may build many `PINNSettings` (one per sweep
value, one per region).

- `representation` — `:power_series` or `:eigenvalue`
- `N`              — power-series degree (ignored by the eigenvalue form's
                     output width, but still sets the supervised target length)
- `neuron_count`   — hidden width
- `maxiters`       — Adam iterations per model
- `n_per_region`   — ODEs sampled per region
- `num_points`     — collocation points
- `seed`           — base seed; each model derives a distinct seed from it
"""
struct ExperimentConfig
  representation::Symbol
  N::Int
  neuron_count::Int
  maxiters::Int
  n_per_region::Int
  num_points::Int
  x_left::Float32
  x_right::Float32
  supervised_weight::Float32
  pde_weight::Float32
  seed::Int
end

function ExperimentConfig(; representation::Symbol=:power_series, N::Int=20,
                            neuron_count::Int=64, maxiters::Int=3000,
                            n_per_region::Int=200, num_points::Int=50,
                            x_left::Float32=0.0f0, x_right::Float32=1.0f0,
                            supervised_weight::Float32=1.0f0,
                            pde_weight::Float32=1.0f0,
                            seed::Int=1234)
  return ExperimentConfig(representation, N, neuron_count, maxiters, n_per_region,
                          num_points, x_left, x_right, supervised_weight,
                          pde_weight, seed)
end

# Build PINNSettings for a dataset under a config. `seed_offset` keeps the
# models in a sweep or transfer experiment independently initialized.
function settings_for(cfg::ExperimentConfig, dataset::Dict;
                      seed_offset::Int=0, neuron_count::Union{Int,Nothing}=nothing,
                      N::Union{Int,Nothing}=nothing)
  Nv = something(N, cfg.N)
  xs = collect(range(cfg.x_left, cfg.x_right, length=cfg.num_points))
  return PINNSettings(
    something(neuron_count, cfg.neuron_count),
    cfg.seed + seed_offset,
    dataset,
    cfg.maxiters,
    Nv, Nv + 1, cfg.num_points,
    cfg.x_left, cfg.x_right,
    cfg.supervised_weight, cfg.pde_weight,
    xs, "adam", cfg.representation,
  )
end

# Batched buffers for a dataset, matching the settings' representation.
function buffers_for(settings::PINNSettings, dataset::Dict)
  items = collect(dataset)
  return settings.representation === :eigenvalue ?
    LF.precompute_batch_eig_buffers(settings, items, false, identity) :
    LF.precompute_batch_buffers(settings, items, false, identity)
end

"""
    evaluate_on(p, net, st, settings, dataset) → NamedTuple

Evaluate trained parameters against a dataset the model may never have seen.

Returns `(pde, bc, sup, relerr)` where `relerr` is SOLUTION-space relative L2 —
the basis-independent metric, so power-series and eigenvalue numbers are
directly comparable. This is the single evaluation primitive behind every
transfer matrix and sweep curve.
"""
function evaluate_on(p, net, st, settings::PINNSettings, dataset::Dict)
  buf = buffers_for(settings, dataset)
  out = first(net(buf.X, p, st))
  pde, bc, sup = settings.representation === :eigenvalue ?
    LF.batched_eigenvalue_losses(out, buf) :
    LF.batched_power_series_losses(out, buf)
  relerr = LF.relative_l2(LF.batched_reconstruct(out, buf), LF.true_solutions(buf))
  return (pde=Float32(pde), bc=Float32(bc), sup=Float32(sup), relerr=relerr)
end

# Train one model on a dataset, returning params + the loss history.
# History is captured per iteration so Shape A can chart convergence.
function train_one(cfg::ExperimentConfig, dataset::Dict, tag::String;
                   seed_offset::Int=0, neuron_count=nothing, N=nothing,
                   output_root::String="results")
  settings = settings_for(cfg, dataset; seed_offset=seed_offset,
                          neuron_count=neuron_count, N=N)
  dir = joinpath(output_root, "exp-$tag")
  mkpath(dir)
  p, net, st, _, _ = train_pinn(settings, dir; run_id=tag,
                                write_loss_csv=true, snapshot_epoch_interval=0)
  PINN.SafeTensorSnapshots.save_safetensors_model(joinpath(dir, "model.safetensors"), p, net, settings.seed;
    extra_metadata=Dict{String,Any}(
      "representation" => String(cfg.representation),
      "objective_components" => "pde + supervised",
      "diagnostic_components" => "bc"
    ))
  return (p=p, net=net, st=st, settings=settings, dir=dir)
end

# Read back the loss history train_pinn wrote, as (iters, total, bc, pde, sup).
function read_loss_history(dir::String)
  path = joinpath(dir, "loss.csv")
  isfile(path) || return nothing
  lines = readlines(path)
  length(lines) < 2 && return nothing
  iters = Float64[]; tot = Float64[]; bc = Float64[]; pde = Float64[]; sup = Float64[]
  for ln in lines[2:end]
    f = split(ln, ",")
    length(f) < 5 && continue
    push!(iters, parse(Float64, f[1])); push!(tot, parse(Float64, f[2]))
    push!(bc, parse(Float64, f[3])); push!(pde, parse(Float64, f[4]))
    push!(sup, parse(Float64, f[5]))
  end
  return (iters=iters, total=tot, bc=bc, pde=pde, sup=sup)
end

# ---------------------------------------------------------------------------
# Shape A — single run
# ---------------------------------------------------------------------------

"""
    run_single(cfg, dataset, name; test_dataset = nothing) → PanelSet

Train one model and chart it: solution comparison, pointwise error, the loss
history, and the per-component breakdown.

`test_dataset` is optional held-out data; when given, the solution panel shows
the model's behaviour on it rather than on the training set.
"""
function run_single(cfg::ExperimentConfig, dataset::Dict, name::String;
                    test_dataset::Union{Dict,Nothing}=nothing,
                    output_root::String="results")
  r = train_one(cfg, dataset, "$(name)-$(cfg.representation)"; output_root=output_root)
  eval_set = something(test_dataset, dataset)
  buf = buffers_for(r.settings, eval_set)
  out = first(r.net(buf.X, r.p, r.st))
  U = LF.batched_reconstruct(out, buf)
  UT = LF.true_solutions(buf)
  xs = collect(range(cfg.x_left, cfg.x_right, length=cfg.num_points))

  # First few ODEs, so the solution panel stays legible.
  show_n = min(3, size(U, 2))
  sol_series = Any[]
  err_series = Any[]
  for b in 1:show_n
    push!(sol_series, ("true $b", xs, Vector(UT[:, b])))
    push!(sol_series, ("pred $b", xs, Vector(U[:, b])))
    push!(err_series, ("|err| $b", xs, abs.(Vector(U[:, b]) .- Vector(UT[:, b]))))
  end

  panels = Panel[
    lines_panel("solution", "Solution comparison ($(cfg.representation))",
                sol_series; xlabel="x", ylabel="u(x)"),
    lines_panel("solution_error", "Absolute error of solution",
                err_series; xlabel="x", ylabel="|u_pred - u_true|", log_y=true),
  ]

  h = read_loss_history(r.dir)
  if h !== nothing
    push!(panels, lines_panel("loss_total", "Total loss",
            [("total", h.iters, h.total)]; xlabel="iteration", ylabel="loss", log_y=true))
    push!(panels, lines_panel("loss_components", "Loss components",
            [("pde", h.iters, h.pde), ("bc", h.iters, h.bc), ("supervised", h.iters, h.sup)];
            xlabel="iteration", ylabel="loss", log_y=true))
  end

  m = evaluate_on(r.p, r.net, r.st, r.settings, eval_set)
  return PanelSet("$name ($(cfg.representation))",
    Dict{String,Any}("representation" => String(cfg.representation),
                     "shape" => "single", "n_odes" => length(dataset),
                     "relerr" => m.relerr, "run_dir" => r.dir),
    panels)
end

# ---------------------------------------------------------------------------
# Shape B — 1-D sweep
# ---------------------------------------------------------------------------

"""
    run_sweep(cfg, dataset, param, values, name) → PanelSet

Train one model per value of `param` and chart the metric against it.

`param` is `:neuron_count` or `:N` — the two things jeet's capacity scripts
sweep (`width_sweep_memorize.jl`, `n_sweep.jl`). Each model gets its own seed
offset so the curve reflects the parameter, not shared initialization luck.
"""
function run_sweep(cfg::ExperimentConfig, dataset::Dict, param::Symbol,
                   values::Vector{Int}, name::String; output_root::String="results")
  param in (:neuron_count, :N) || error(
    "run_sweep: param must be :neuron_count or :N, got :$param")

  relerrs = Float64[]; pdes = Float64[]; bcs = Float64[]; sups = Float64[]
  for (i, v) in enumerate(values)
    kw = param === :neuron_count ? (neuron_count=v, N=nothing) : (neuron_count=nothing, N=v)
    r = train_one(cfg, dataset, "$(name)-$(cfg.representation)-$(param)$(v)";
                  seed_offset=i, kw..., output_root=output_root)
    m = evaluate_on(r.p, r.net, r.st, r.settings, dataset)
    push!(relerrs, m.relerr); push!(pdes, m.pde); push!(bcs, m.bc); push!(sups, m.sup)
  end

  xv = Float64.(values)
  panels = Panel[
    lines_panel("sweep_relerr", "Solution rel-L2 vs $param ($(cfg.representation))",
                [("rel L2", xv, relerrs)];
                xlabel=String(param), ylabel="relative L2", log_y=true),
    lines_panel("sweep_components", "Loss components vs $param",
                [("pde", xv, pdes), ("bc", xv, bcs), ("supervised", xv, sups)];
                xlabel=String(param), ylabel="loss component", log_y=true),
  ]

  return PanelSet("$name sweep over $param ($(cfg.representation))",
    Dict{String,Any}("representation" => String(cfg.representation),
                     "shape" => "sweep", "param" => String(param),
                     "values" => values),
    panels)
end

# ---------------------------------------------------------------------------
# Shape C — cross-product / transfer
# ---------------------------------------------------------------------------

"""
    run_transfer(cfg, train_regions, test_regions, name) → PanelSet

Train one model per region in `train_regions`, then evaluate every model
against every region in `test_regions`.

Produces the transfer matrices: entry (i, j) is the metric for "trained on i,
tested on j". The diagonal is in-family (expected low); off-diagonal is
out-of-family (expected high). Also emits the per-component matrices and the
in-family vs out-of-family bar summary.

Cost: `length(train_regions)` models. With batched losses this is minutes
rather than hours.
"""
function run_transfer(cfg::ExperimentConfig, train_regions::Vector{Symbol},
                      test_regions::Vector{Symbol}, name::String;
                      output_root::String="results", rng_seed::Int=1234)
  nr, nc = length(train_regions), length(test_regions)

  # Test sets are fixed up front so every model is judged on identical data.
  test_sets = Dict{Symbol,Dict}()
  for (j, reg) in enumerate(test_regions)
    test_sets[reg] = generate_region_dataset(reg, cfg.n_per_region, cfg.N;
                                             rng=MersenneTwister(rng_seed + 1000 + j))
  end

  M_rel = zeros(nr, nc); M_pde = zeros(nr, nc)
  M_bc = zeros(nr, nc);  M_sup = zeros(nr, nc)

  for (i, treg) in enumerate(train_regions)
    train_set = generate_region_dataset(treg, cfg.n_per_region, cfg.N;
                                        rng=MersenneTwister(rng_seed + i))
    r = train_one(cfg, train_set, "$(name)-$(cfg.representation)-train_$(treg)";
                  seed_offset=i, output_root=output_root)
    for (j, tereg) in enumerate(test_regions)
      m = evaluate_on(r.p, r.net, r.st, r.settings, test_sets[tereg])
      M_rel[i, j] = m.relerr; M_pde[i, j] = m.pde
      M_bc[i, j] = m.bc;      M_sup[i, j] = m.sup
    end
  end

  rl = String.(train_regions); cl = String.(test_regions)
  panels = Panel[
    heatmap_panel("transfer_relerr", "Solution rel-L2 transfer ($(cfg.representation))",
                  M_rel, rl, cl; xlabel="tested on", ylabel="trained on"),
    heatmap_panel("transfer_pde", "PDE residual component (raw)",
                  M_pde, rl, cl; xlabel="tested on", ylabel="trained on"),
    heatmap_panel("transfer_bc", "IC component (raw)",
                  M_bc, rl, cl; xlabel="tested on", ylabel="trained on"),
    heatmap_panel("transfer_sup", "Supervised component (raw)",
                  M_sup, rl, cl; xlabel="tested on", ylabel="trained on"),
  ]

  # In-family (diagonal) vs out-of-family (off-diagonal), geometric mean —
  # geometric because these components span orders of magnitude.
  geo(v) = isempty(v) ? 0.0 : exp(mean(log.(max.(v, 1e-20))))
  diagvals(M) = [M[i, i] for i in 1:min(nr, nc)]
  offvals(M) = [M[i, j] for i in 1:nr, j in 1:nc if i != j]
  in_fam = [geo(diagvals(M)) for M in (M_pde, M_bc, M_sup)]
  out_fam = [geo(offvals(M)) for M in (M_pde, M_bc, M_sup)]
  ratios = ["", "", ""]
  ratios = [in_fam[k] > 0 ? "x$(round(out_fam[k]/in_fam[k]; sigdigits=2))" : ""
            for k in 1:3]

  push!(panels, grouped_bar_panel("component_bars",
    "Loss components: in-family vs out-of-family ($(cfg.representation))",
    ["PDE residual", "IC", "supervised"],
    [("in-family (diagonal)", in_fam), ("out-of-family (off-diagonal)", out_fam)];
    ylabel="loss component (raw, geometric mean)",
    annotations=[["", "", ""], ratios]))

  return PanelSet("$name transfer ($(cfg.representation))",
    Dict{String,Any}("representation" => String(cfg.representation),
                     "shape" => "transfer",
                     "train_regions" => rl, "test_regions" => cl,
                     "n_per_region" => cfg.n_per_region),
    panels)
end

# ===========================================================================
# Shape D — extrapolation sweep
# ===========================================================================

"""
    run_extrapolate(cfg, regions, Rmax, name) → PanelSet

Per-region extrapolation: train in-box, test on concentric shells R=1..Rmax
of the **same** region. Produces one error-vs-R curve per region, arranged
in a grid.
"""
function run_extrapolate(cfg::ExperimentConfig, regions::Vector{Symbol},
                         Rmax::Int, name::String;
                         output_root::String="results", rng_seed::Int=1234)
  n_reg = length(regions)
  shell_n = max(5, cfg.n_per_region ÷ 4)  # fewer ODEs per shell (fast eval)
  rng = MersenneTwister(rng_seed)

  # Pre-build shell datasets once — shared across all models
  shell_sets = Dict{Tuple{Symbol,Int},Dict}()
  for reg in regions, R in 1:Rmax
    shell_sets[(reg, R)] = generate_shell_dataset(reg, R, shell_n, cfg.N; rng=MersenneTwister(rng_seed + 1000 + R))
  end

  # Train one model per region, evaluate on its own shells
  curves = Dict{Symbol,Vector{Float32}}()
  for reg in regions
    train_set = generate_region_dataset(reg, cfg.n_per_region, cfg.N; rng=MersenneTwister(rng_seed + findfirst(==(reg), regions)))
    r = train_one(cfg, train_set, "$(name)-$(cfg.representation)-$(reg)";
                  seed_offset=findfirst(==(reg), regions), output_root=output_root)
    errs = Float32[]
    for R in 1:Rmax
      m = evaluate_on(r.p, r.net, r.st, r.settings, shell_sets[(reg, R)])
      push!(errs, m.relerr)
    end
    curves[reg] = errs
  end

  xvec = collect(1:Rmax)
  series = [(String(reg), xvec, collect(curves[reg])) for reg in regions]
  lines_panels = [lines_panel("extrap_curves",
    "Extrapolation: error vs R ($(cfg.representation))",
    series; xlabel="shell radius R", ylabel="solution rel-L2",
    log_y=true)]

  return PanelSet("$name extrapolation ($(cfg.representation))",
    Dict{String,Any}("representation" => String(cfg.representation),
                     "shape" => "extrapolate", "Rmax" => Rmax,
                     "regions" => String.(regions)),
    lines_panels)
end

# ===========================================================================
# Shape E — cross-region range (family × range combined)
# ===========================================================================

"""
    run_cross_region_range(cfg, train_regions, test_regions, Rmax, name) → PanelSet

For each training region, test on every test region at concentric shells
R=1..Rmax. Produces one panel per training region: one error-vs-R line per
test region (solid = in-family, dashed = out-of-family).
"""
function run_cross_region_range(cfg::ExperimentConfig,
                                train_regions::Vector{Symbol},
                                test_regions::Vector{Symbol},
                                Rmax::Int, name::String;
                                output_root::String="results",
                                rng_seed::Int=1234)
  shell_n = max(5, cfg.n_per_region ÷ 4)
  rng = MersenneTwister(rng_seed)

  # Pre-build all shell datasets
  shell_sets = Dict{Tuple{Symbol,Int},Dict}()
  for reg in test_regions, R in 1:Rmax
    shell_sets[(reg, R)] = generate_shell_dataset(reg, R, shell_n, cfg.N; rng=MersenneTwister(rng_seed + 2000 + R))
  end

  panels = Panel[]
  for (i, treg) in enumerate(train_regions)
    train_set = generate_region_dataset(treg, cfg.n_per_region, cfg.N; rng=MersenneTwister(rng_seed + i))
    r = train_one(cfg, train_set, "$(name)-$(cfg.representation)-train_$(treg)";
                  seed_offset=i, output_root=output_root)
    series = []
    styles = String[]
    for tereg in test_regions
      errs = Float32[]
      for R in 1:Rmax
        m = evaluate_on(r.p, r.net, r.st, r.settings, shell_sets[(tereg, R)])
        push!(errs, m.relerr)
      end
      push!(series, (String(tereg), collect(1:Rmax), collect(errs)))
      push!(styles, tereg === treg ? "solid" : "dash")
    end
    push!(panels, lines_panel("range_$(treg)",
      "trained on $(treg) — error vs R",
      series; xlabel="shell radius R",
      ylabel="solution rel-L2", log_y=true, linestyles=styles))
  end

  return PanelSet("$name cross-region range ($(cfg.representation))",
    Dict{String,Any}("representation" => String(cfg.representation),
                     "shape" => "cross_region_range", "Rmax" => Rmax,
                     "train_regions" => String.(train_regions),
                     "test_regions" => String.(test_regions)),
    panels)
end

# ===========================================================================
# Shape F — showcase (solution curves for test scenarios)
# ===========================================================================

"""
    run_showcase(cfg, name) → PanelSet

Train on the exponential family (saddle, stable_node, unstable_node) in-box,
then test 6 scenarios and produce solution-curve overlays + grouped bar of
rel-L2 per scenario.
"""
function run_showcase(cfg::ExperimentConfig, name::String;
                      output_root::String="results", rng_seed::Int=1234)
  exp_regions = [:saddle, :stable_node, :unstable_node]
  rng = MersenneTwister(rng_seed)

  train_set = Dict{Any,Any}()
  for reg in exp_regions
    merge!(train_set, generate_region_dataset(reg, cfg.n_per_region ÷ 3, cfg.N;
              rng=MersenneTwister(rng_seed + findfirst(==(reg), exp_regions))))
  end

  r = train_one(cfg, train_set, "$(name)-$(cfg.representation)"; output_root=output_root)

  # 6 test scenarios
  scenarios = [
    ("memorize",     (τ=0.5f0,   Δ=-0.5f0)),     # in-train saddle
    ("in-family",    (τ=1.2f0,   Δ=0.3f0)),      # held-out stable_node
    ("out-range A",  (τ=2.5f0,   Δ=-1.0f0)),     # saddle beyond box
    ("out-range B",  (τ=-2.5f0,  Δ=1.5f0)),      # node beyond box
    ("out-family C", (τ=0.5f0,   Δ=1.0f0)),      # stable_spiral
    ("out-family D", (τ=0.0f0,   Δ=1.0f0)),      # center
  ]

  xs = collect(range(cfg.x_left, cfg.x_right, length=cfg.num_points))
  sol_curves = []
  bar_data = Float32[]

  for (label, (; τ, Δ)) in scenarios
    ds = generate_grid_dataset([τ], [Δ], cfg.N)
    buf = buffers_for(r.settings, ds)
    out = first(r.net(buf.X, r.p, r.st))
    U = LF.batched_reconstruct(out, buf)
    UT = LF.true_solutions(buf)
    err = LF.relative_l2(U[:, 1], UT[:, 1])
    push!(bar_data, err)
    push!(sol_curves, ("$label true", xs, collect(UT[:, 1])))
    push!(sol_curves, ("$label pred", xs, collect(U[:, 1])))
  end

  scenario_names = [s[1] for s in scenarios]

  curve_panels = Panel[
    lines_panel("showcase_curves", "Solution curves ($(cfg.representation))",
      sol_curves;
      xlabel="x", ylabel="u(x)"),
  ]

  push!(curve_panels,
    grouped_bar_panel("showcase_bars",
      "Solution rel-L2 per scenario ($(cfg.representation))",
      scenario_names,
      [("rel-L2", collect(bar_data))];
      ylabel="solution rel-L2", log_y=true))

  return PanelSet("$name showcase ($(cfg.representation))",
    Dict{String,Any}("representation" => String(cfg.representation),
                     "shape" => "showcase"),
    curve_panels)
end

# ===========================================================================
# Shape G — generalization radius
# ===========================================================================

"""
    run_gen_radius(cfg, name; mode) → PanelSet

Generalization-radius experiment. Two modes:
  :disk   — train on compact disk of (τ,Δ), evaluate on full plane grid
  :family — train one model per region, map area under ε-contour
"""
function run_gen_radius(cfg::ExperimentConfig, name::String;
                        mode::Symbol=:disk,
                        output_root::String="results",
                        rng_seed::Int=1234,
                        eps_tol::Float32=0.10f0,
                        Lmap::Float32=4.0f0, Ng::Int=41)
  rng = MersenneTwister(rng_seed)
  panels = Panel[]

  if mode === :disk
    # Train on compact disk centered at (τ=0, Δ=0), radius R_train
    cτ, cΔ = 0.0f0, 0.0f0
    R_train = 1.5f0

    # Build training set by rejection sampling on a disk
    train_taus = Float32[]; train_deltas = Float32[]
    while length(train_taus) < cfg.n_per_region
      t = rand(rng, Float32) * 2R_train - R_train
      d = rand(rng, Float32) * 2R_train - R_train
      if t^2 + d^2 <= R_train^2
        push!(train_taus, t); push!(train_deltas, d)
      end
    end
    train_set = generate_grid_dataset(train_taus, train_deltas, cfg.N)

    r = train_one(cfg, train_set, "$(name)-$(cfg.representation)-disk"; output_root=output_root)

    # Build grid evaluation set
    grid_ts = Float32[]; grid_ds = Float32[]
    for i in 1:Ng, j in 1:Ng
      t = -Lmap + (2Lmap / (Ng - 1)) * (i - 1)
      d = -Lmap + (2Lmap / (Ng - 1)) * (j - 1)
      push!(grid_ts, Float32(t)); push!(grid_ds, Float32(d))
    end
    grid_set = generate_grid_dataset(grid_ts, grid_ds, cfg.N)
    buf = buffers_for(r.settings, grid_set)
    out = first(r.net(buf.X, r.p, r.st))
    U_grid = LF.batched_reconstruct(out, buf)
    UT_grid = LF.true_solutions(buf)

    # Build error matrix
    err_map = zeros(Float32, Ng, Ng)
    for idx in 1:length(grid_ts)
      i = (idx - 1) % Ng + 1
      j = (idx - 1) ÷ Ng + 1
      err_map[j, i] = LF.relative_l2(U_grid[:, idx], UT_grid[:, idx])
    end

    # Heatmap panel
    t_axis = range(-Lmap, Lmap, length=Ng)
    d_axis = range(-Lmap, Lmap, length=Ng)
    push!(panels, heatmap_panel("genradius_map",
      "Generalization radius ($(cfg.representation))",
      err_map, string.(round.(d_axis, digits=1)), string.(round.(t_axis, digits=1));
      xlabel="τ", ylabel="Δ", log_color=true, annotate=false,
      x_values=t_axis, y_values=d_axis, contour_levels=[eps_tol]))

  elseif mode === :family
    regions = [:saddle, :stable_node, :unstable_node,
               :stable_spiral, :unstable_spiral, :center]
    areas = Float32[]

    t_axis = range(-Lmap, Lmap, length=Ng)
    d_axis = range(-Lmap, Lmap, length=Ng)

    for (i, reg) in enumerate(regions)
      train_set = generate_region_dataset(reg, cfg.n_per_region, cfg.N;
                                          rng=MersenneTwister(rng_seed + i))
      rr = train_one(cfg, train_set, "$(name)-$(cfg.representation)-family_$(reg)";
                     seed_offset=i, output_root=output_root)

      # Build grid evaluation for this region
      grid_ts = Float32[]; grid_ds = Float32[]
      for i2 in 1:Ng, j2 in 1:Ng
        t = -Lmap + (2Lmap / (Ng - 1)) * (i2 - 1)
        d = -Lmap + (2Lmap / (Ng - 1)) * (j2 - 1)
        push!(grid_ts, Float32(t)); push!(grid_ds, Float32(d))
      end
      grid_set = generate_grid_dataset(grid_ts, grid_ds, cfg.N)
      buf = buffers_for(rr.settings, grid_set)
      out = first(rr.net(buf.X, rr.p, rr.st))
      U_grid = LF.batched_reconstruct(out, buf)
      UT_grid = LF.true_solutions(buf)

      # Compute area under ε-contour
      cell_area = (2Lmap / (Ng - 1))^2
      area_under = 0.0f0
      err_grid = zeros(Float32, Ng, Ng)
      for idx in 1:length(grid_ts)
        i2 = (idx - 1) % Ng + 1; j2 = (idx - 1) ÷ Ng + 1
        e = LF.relative_l2(U_grid[:, idx], UT_grid[:, idx])
        err_grid[j2, i2] = e
        if e < eps_tol
          area_under += cell_area
        end
      end
      push!(areas, area_under)

      push!(panels, heatmap_panel("genradius_family_$(reg)",
        "Error map — $(reg) ($(cfg.representation))",
        err_grid, string.(round.(d_axis, digits=1)), string.(round.(t_axis, digits=1));
        xlabel="τ", ylabel="Δ", log_color=true, annotate=false,
        x_values=t_axis, y_values=d_axis, contour_levels=[eps_tol]))
    end

    # Bar chart of area per family
    push!(panels, grouped_bar_panel("genradius_areas",
      "Generalization area (< ε) per family ($(cfg.representation))",
      String.(regions),
      [("area", collect(areas))];
      ylabel="area where rel-L2 < $(eps_tol)"))
  else
    error("Unknown gen_radius mode: :$(mode). Expected :disk or :family.")
  end

  return PanelSet("$name gen-radius $(mode) ($(cfg.representation))",
    Dict{String,Any}("representation" => String(cfg.representation),
                     "shape" => "gen_radius", "mode" => String(mode),
                     "eps_tol" => eps_tol, "Lmap" => Lmap, "Ng" => Ng),
    panels)
end

end # module Experiments
