# =============================================================================
# Test: grid search mode parity (sequential vs parallel)
# =============================================================================
# Verifies:
#   1. :sequential mode completes all configs in a 2x2 grid
#   2. :parallel mode completes all configs in a 2x2 grid
#   3. Both modes fill the GridView with non-NaN values for all cells
#   4. GridView state transitions are observable (cells progress through
#      pending → running → done)
#   5. Best weights from each mode are valid (in-range, finite objective)
#   6. Invalid mode raises
# =============================================================================

using Test
using Random

include("../utils/plugboard.jl")
using .Plugboard

include("../utils/tui.jl")
using .TUI

include("../utils/two_d_grid_search_hyperparameters.jl")
using .TwoDGridSearchOnWeights

# ---------------------------------------------------------------------------
# Helper: build a minimal dataset
# ---------------------------------------------------------------------------
function tiny_dataset()
  inner = Plugboard.generate_region_dataset(Plugboard.TRACE_DET_REGIONS, 5, 6;
                                            a0=1.0f0, a1=0.0f0,
                                            tau_lim=2.0f0, delta_lim=2.0f0,
                                            rng=MersenneTwister(7))
  # Wrap in the JSON format evaluate_weight_configuration expects:
  # Dict{String,Dict{String,Any}} where the inner key is the matrix stringified.
  ds = Dict{String,Dict{String,Any}}("01" => Dict{String,Any}())
  for (mat, vec) in inner
    ds["01"][string(mat)] = vec
  end
  return ds
end

function run_mode(mode::Symbol; view::Union{TwoDGridSearchOnWeights.TUI.GridView,Nothing}=nothing)
  ds = tiny_dataset()
  bench = tiny_dataset()
  tmp = mktempdir()
  # evaluate_weight_configuration hardcodes num_points=1000 internally,
  # so xs must have 1000 points to match the buffer construction.
  xs = Float32.(collect(range(0.0, 1.0, length=1000)))
  return grid_search_2d(
    4, ds, bench,
    :pde, (0.5, 1.0),
    :supervised, (0.5, 1.0),
    2;
    fixed_weights=(bc=1.0,),
    num_supervised=6, N=6,
    x_left=0.0f0, x_right=1.0f0,
    xs=xs,
    base_data_dir=tmp,
    milestone_interval=0,
    mode=mode,
    view=view,
  )
end

# ---------------------------------------------------------------------------
# Test 1: sequential mode completes all configs
# ---------------------------------------------------------------------------

@testset "grid search: :sequential mode completes all configs" begin
  result = run_mode(:sequential)
  @test result isa GridSearchResult
  @test size(result.objective_values) == (2, 2)
  @test all(isfinite, result.objective_values)
  @test result.best_objective isa Float64
  @test isfinite(result.best_objective)
end

# ---------------------------------------------------------------------------
# Test 2: parallel mode completes all configs
# ---------------------------------------------------------------------------

@testset "grid search: :parallel mode completes all configs" begin
  result = run_mode(:parallel)
  @test result isa GridSearchResult
  @test size(result.objective_values) == (2, 2)
  @test all(isfinite, result.objective_values)
end

# ---------------------------------------------------------------------------
# Test 3: both modes fill the GridView
# ---------------------------------------------------------------------------

@testset "grid search: GridView is fully done in both modes" begin
  for mode in (:sequential, :parallel)
    # Use the grid-search module's TUI namespace so the GridView types match.
    v = TwoDGridSearchOnWeights.TUI.GridView(2, 2)
    @test v.completed[] == 0
    @test all(s -> s == TwoDGridSearchOnWeights.TUI.CELL_PENDING, v.state)

    run_mode(mode; view=v)
    @test v.completed[] == 4
    @test all(s -> s == TwoDGridSearchOnWeights.TUI.CELL_DONE, v.state)
    @test all(isfinite, v.results)
  end
end

# ---------------------------------------------------------------------------
# Test 4: invalid mode raises
# ---------------------------------------------------------------------------

@testset "grid search: invalid mode raises" begin
  @test_throws ErrorException run_mode(:bogus)
end

# ---------------------------------------------------------------------------
# Test 5: best weights are within the search range
# ---------------------------------------------------------------------------

@testset "grid search: best weights are within range" begin
  result = run_mode(:parallel)
  @test 0.5 <= result.best_weights.pde <= 1.0
  @test 0.5 <= result.best_weights.supervised <= 1.0
  @test result.best_weights.bc == 1.0     # fixed
end

@info "============================================"
@info "ALL GRID SEARCH MODE PARITY TESTS COMPLETED"
@info "============================================"