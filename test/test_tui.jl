# =============================================================================
# Test: TUI module (GPUBoard + GridView)
# =============================================================================
# Verifies:
#   1. GPUBoard construction + update + slot transitions
#   2. GridView construction + state transitions (mark_running/done/failed)
#   3. Build_bar produces the expected glyph widths
#   4. is_tty() respects JULIA_TUI_OFF and CI environment overrides
#   5. Concurrent updates from many tasks converge to the final state
#   6. notty path never emits ANSI escape codes
# =============================================================================

using Test

include("../utils/tui.jl")
using .TUI

# ---------------------------------------------------------------------------
# Test 1: GPUBoard construction + initial state
# ---------------------------------------------------------------------------

@testset "GPUBoard: construction + initial state" begin
  devices = ["GPU 0 (A100)", "GPU 1 (A100)", "GPU 2 (A100)"]
  board = TUI.GPUBoard(devices; max_iter=1000)

  @test length(board.devices) == 3
  @test length(board.variants) == 3
  @test all(v -> isempty(v), board.variants)         # all idle initially
  @test all(it -> it == 0, board.iter)
  @test all(m -> m == 1000, board.max_iter)
  @test board.first_render == false                  # first render cleared it
end

# ---------------------------------------------------------------------------
# Test 2: GPUBoard update! — variant change transitions
# ---------------------------------------------------------------------------

@testset "GPUBoard: update! transitions variant + iter + loss" begin
  board = TUI.GPUBoard(["GPU 0", "GPU 1"]; max_iter=500)

  TUI.update!(board, 1; variant="ps_N20", iter=100, loss=0.001f0)
  @test board.variants[1] == "ps_N20"
  @test board.iter[1] == 100
  @test board.loss[1] ≈ 0.001f0
  @test board.variants[2] == ""                     # other slot untouched

  # Partial update: just iter+loss, no variant
  TUI.update!(board, 1; iter=200, loss=0.0005f0)
  @test board.variants[1] == "ps_N20"               # unchanged
  @test board.iter[1] == 200
  @test board.loss[1] ≈ 0.0005f0

  # Variant change
  TUI.update!(board, 2; variant="ps_N25", iter=50, loss=0.002f0)
  @test board.variants[2] == "ps_N25"
  @test board.iter[2] == 50

  # Idle slot remains idle
  @test board.variants[2] != ""                     # slot 2 was just updated
end

# ---------------------------------------------------------------------------
# Test 3: GPUBoard max_iter override
# ---------------------------------------------------------------------------

@testset "GPUBoard: max_iter override" begin
  board = TUI.GPUBoard(["GPU 0"]; max_iter=1000)
  TUI.update!(board, 1; variant="v", iter=10, max_iter=200, loss=0.1f0)
  @test board.max_iter[1] == 200
end

# ---------------------------------------------------------------------------
# Test 4: build_bar glyph widths
# ---------------------------------------------------------------------------

@testset "build_bar: width and progress" begin
  @test TUI.build_bar(0, 100)   == "[" * " "^20 * "]"
  @test TUI.build_bar(100, 100) == "[" * "█"^20 * "]"
  @test TUI.build_bar(50, 100)  == "[" * "█"^10 * " "^10 * "]"
  @test TUI.build_bar(0, 0)     == "[" * " "^20 * "]"

  # Custom width
  bar10 = TUI.build_bar(50, 100; width=10)
  @test length(bar10) == 10 + 2                      # 10 glyphs + 2 brackets
  @test count(==('█'), bar10) == 5
end

# ---------------------------------------------------------------------------
# Test 5: GridView construction + state transitions
# ---------------------------------------------------------------------------

@testset "GridView: state transitions" begin
  v = TUI.GridView(3, 4)
  @test v.rows == 3
  @test v.cols == 4
  @test v.total == 12
  @test v.completed[] == 0
  @test all(s -> s == TUI.CELL_PENDING, v.state)    # all pending initially

  TUI.mark_running!(v, 1, 1)
  @test v.state[1, 1] == TUI.CELL_RUNNING

  TUI.mark_done!(v, 1, 1, 0.001)
  @test v.state[1, 1] == TUI.CELL_DONE
  @test v.results[1, 1] ≈ 0.001
  @test v.completed[] == 1

  TUI.mark_failed!(v, 2, 3)
  @test v.state[2, 3] == TUI.CELL_FAILED

  # Other cells remain pending
  @test v.state[3, 4] == TUI.CELL_PENDING
end

# ---------------------------------------------------------------------------
# Test 6: GridView completed counter is thread-safe
# ---------------------------------------------------------------------------

@testset "GridView: concurrent mark_done!" begin
  v = TUI.GridView(4, 4)
  @test v.total == 16

  tasks = Task[]
  for i in 1:4, j in 1:4
    push!(tasks, Threads.@spawn TUI.mark_done!(v, i, j, Float64(i * 10 + j)))
  end
  foreach(fetch, tasks)

  @test v.completed[] == 16
  @test all(s -> s == TUI.CELL_DONE, v.state)
end

# ---------------------------------------------------------------------------
# Test 7: is_tty respects JULIA_TUI_OFF
# ---------------------------------------------------------------------------

@testset "is_tty: environment overrides" begin
  # Baseline: CI=unset, JULIA_TUI_OFF=unset → depends on stdout; just ensure
  # the function returns a Bool (no exception).
  @test TUI.is_tty() isa Bool

  # JULIA_TUI_OFF=1 forces false
  withenv("JULIA_TUI_OFF" => "1") do
    @test TUI.is_tty() == false
  end

  # CI=1 forces false
  withenv("CI" => "1", "JULIA_TUI_OFF" => "") do
    @test TUI.is_tty() == false
  end
end

# ---------------------------------------------------------------------------
# Test 8: notty fallback does not emit ANSI escape codes
# ---------------------------------------------------------------------------

@testset "notty fallback: no ANSI in plain-text path" begin
  # Force the plain path
  withenv("JULIA_TUI_OFF" => "1", "CI" => "1") do
    # The render! functions still run but should not emit ANSI; they emit @info
    # lines. We can't capture @info easily, but we can check that is_tty()
    # returns false and that the field is set.
    board = TUI.GPUBoard(["GPU 0"]; max_iter=10)
    @test board.tty == false
    v = TUI.GridView(2, 2)
    @test v.tty == false

    # Update paths should not throw under notty
    TUI.update!(board, 1; variant="x", iter=5, loss=0.5f0)
    TUI.mark_running!(v, 1, 1)
    TUI.mark_done!(v, 1, 1, 1.0)
    @test board.variants[1] == "x"
    @test v.state[1, 1] == TUI.CELL_DONE
  end
end

# ---------------------------------------------------------------------------
# Test 9: GPUBoard concurrent updates from many tasks converge
# ---------------------------------------------------------------------------

@testset "GPUBoard: concurrent updates converge" begin
  board = TUI.GPUBoard(["GPU 0", "GPU 1", "GPU 2", "GPU 3"]; max_iter=100)

  tasks = Task[]
  for slot in 1:4
    for k in 1:25
      push!(tasks, Threads.@spawn begin
        TUI.update!(board, slot;
                    iter=k,
                    loss=Float32(k) / 1000,
                    variant=k == 1 ? "variant_$slot" : "")
      end)
    end
  end
  foreach(fetch, tasks)

  # Variant name: the first update on each slot wins, subsequent updates with
  # an empty variant must NOT clobber it.
  @test all(i -> board.variants[i] == "variant_$i", 1:4)

  # iter: every update sets it (1..25), so the final value must be in [1, 25].
  # Last-write-wins is order-dependent under threads, so we don't assert ==25.
  @test all(i -> 1 <= board.iter[i] <= 25, 1:4)
end

@info "============================================"
@info "ALL TUI TESTS COMPLETED"
@info "============================================"