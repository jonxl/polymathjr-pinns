# =============================================================================
# Test: gen_radius multi-level contour (error spread visualization)
# =============================================================================
# Verifies:
#   1. eps_levels kwarg flows through to heatmap panel contour_levels
#   2. :disk mode produces one heatmap with all eps_levels as contour lines
#   3. :family mode produces N heatmaps (one per region) + a grouped bar with
#      one series per eps_level
#   4. PanelSet metadata exposes the eps_levels vector (replaces old eps_tol)
#   5. Custom eps_levels override the default
#   6. _fmt_eps produces compact decimal labels for ε
# =============================================================================

using Test
using JSON

include("../utils/tui.jl")
using .TUI

include("../viz/PanelSpec.jl")
using .PanelSpec

include("../utils/experiments.jl")
using .Experiments

# ---------------------------------------------------------------------------
# Test 1: _fmt_eps helper
# ---------------------------------------------------------------------------

@testset "_fmt_eps: compact decimal labels" begin
  @test Experiments._fmt_eps(0.0001f0) == "0.0001"
  @test Experiments._fmt_eps(0.001f0)  == "0.001"
  @test Experiments._fmt_eps(0.01f0)   == "0.01"
  @test Experiments._fmt_eps(0.1f0)    == "0.1"
  @test Experiments._fmt_eps(1.0f0)    == "1.0"
  @test Experiments._fmt_eps(10.0f0)   == "10.0"
  @test Experiments._fmt_eps(0.0f0)    == "0"
  # Float32 precision survives rounding to 4 sig digits
  @test Experiments._fmt_eps(0.0001f0) == "0.0001"
end

# ---------------------------------------------------------------------------
# Test 2: :disk mode — multi-level contour on the single heatmap
# ---------------------------------------------------------------------------

@testset "run_gen_radius :disk: multi-level contour on heatmap" begin
  tmp = mktempdir()
  cfg = ExperimentConfig(representation=:power_series, maxiters=8, n_per_region=3)
  ps = run_gen_radius(cfg, "test_disk"; mode=:disk, Ng=5, output_root=tmp)

  @test length(ps.panels) == 1                          # single heatmap

  heat = ps.panels[1]
  @test heat.kind === :heatmap
  @test heat.id == "genradius_map"

  # Contour levels: defaults are 6 levels spanning 1e-4..1e+1
  levels = heat.opts["contour_levels"]
  @test length(levels) == 6
  @test length(levels) == 6
  @test Float64.(levels) ≈ Float64[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
end

# ---------------------------------------------------------------------------
# Test 3: :family mode — multi-level heatmaps + multi-series grouped bar
# ---------------------------------------------------------------------------

@testset "run_gen_radius :family: multi-series bar + per-region heatmaps" begin
  tmp = mktempdir()
  cfg = ExperimentConfig(representation=:power_series, maxiters=6, n_per_region=3)
  ps = run_gen_radius(cfg, "test_family"; mode=:family, Ng=5, output_root=tmp)

  # Six region heatmaps + one grouped bar
  @test length(ps.panels) == 7

  # Every heatmap carries all eps_levels as contour lines
  for p in ps.panels
    if p.kind === :heatmap
      @test haskey(p.opts, "contour_levels")
      @test length(p.opts["contour_levels"]) == 6
    end
  end

  # Find the grouped bar
  bar = nothing
  for p in ps.panels
    if p.kind === :grouped_bar
      bar = p
      break
    end
  end
  @test bar !== nothing
  @test bar.id == "genradius_areas"

  # One series per eps_level — each entry is a Dict{"name", "values"}.
  series_names = [s["name"] for s in bar.data["series"]]
  @test length(series_names) == 6
  @test series_names == ["ε=0.0001", "ε=0.001", "ε=0.01",
                         "ε=0.1", "ε=1.0", "ε=10.0"]
end

# ---------------------------------------------------------------------------
# Test 4: PanelSet metadata exposes eps_levels
# ---------------------------------------------------------------------------

@testset "run_gen_radius: PanelSet metadata has eps_levels (not eps_tol)" begin
  tmp = mktempdir()
  cfg = ExperimentConfig(representation=:power_series, maxiters=6, n_per_region=3)
  ps = run_gen_radius(cfg, "test_meta"; mode=:disk, Ng=5, output_root=tmp)

  @test haskey(ps.meta, "eps_levels")
  @test !haskey(ps.meta, "eps_tol")                   # old key removed
  @test ps.meta["eps_levels"] == Float32[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
end

# ---------------------------------------------------------------------------
# Test 5: custom eps_levels override the default
# ---------------------------------------------------------------------------

@testset "run_gen_radius: custom eps_levels propagate" begin
  tmp = mktempdir()
  cfg = ExperimentConfig(representation=:power_series, maxiters=6, n_per_region=3)
  custom = Float32[0.05, 0.5, 5.0]
  ps = run_gen_radius(cfg, "test_custom"; mode=:disk, Ng=5,
                      output_root=tmp, eps_levels=custom)

  @test ps.meta["eps_levels"] == custom
  @test ps.panels[1].opts["contour_levels"] == custom
end

# ---------------------------------------------------------------------------
# Test 6: end-to-end — roundtrip through JSON
# ---------------------------------------------------------------------------

@testset "run_gen_radius: panelset.json preserves multi-level structure" begin
  tmp = mktempdir()
  cfg = ExperimentConfig(representation=:power_series, maxiters=6, n_per_region=3)
  ps = run_gen_radius(cfg, "test_json"; mode=:family, Ng=5, output_root=tmp)

  out_path = joinpath(tmp, "test_json_panelset.json")
  save_experiment(out_path, ps)
  @test isfile(out_path)

  parsed = JSON.parsefile(out_path)
  # JSON stores eps_levels as Float64 (no Float32 in JSON); compare as floats
  @test Float64.(parsed["meta"]["eps_levels"]) ≈
        Float64[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]

  # Each region's heatmap carries all 6 contour levels
  heatmaps = filter(p -> p["kind"] == "heatmap", parsed["panels"])
  @test length(heatmaps) == 6
  for p in heatmaps
    @test length(p["opts"]["contour_levels"]) == 6
  end

  # Bar chart carries 6 series (one per ε level)
  bar = filter(p -> p["kind"] == "grouped_bar", parsed["panels"])[1]
  @test length(bar["data"]["series"]) == 6
end

@info "============================================"
@info "ALL GEN-RADIUS MULTI-LEVEL CONTOUR TESTS COMPLETED"
@info "============================================"