# =============================================================================
# utils/tui.jl — Terminal UI helpers (GPUBoard + GridView)
#
# Two render targets share one data model and one API:
#   - GPUBoard : static "cashier board" — one row per GPU, persistent.
#                Updates overwrite in place on a TTY; otherwise updates emit
#                plain @info lines only when state actually changes.
#   - GridView : grid-search view — compact colored cells; one cell per
#                (weight1, weight2) configuration. Same in-place / log split.
#
# Opt-in rendering: detects whether stdout is a TTY via is_tty() and falls
# back to plain tagged logs when not. CI, file-redirected runs, and batch
# jobs never see ANSI escapes or in-place overwrites.
#
# Concurrency: every public mutator takes the board's ReentrantLock so a
# worker task that calls update!(...) cannot interleave with another worker's
# render!. update! is cheap (one printf) so contention is not a concern.
# =============================================================================

module TUI

using Base.Threads: Atomic, atomic_add!

export is_tty, GPUBoard, GridView,
       update!, mark_running!, mark_done!, mark_failed!

# ---------------------------------------------------------------------------
# TTY detection
# ---------------------------------------------------------------------------

"""
    is_tty() → Bool

Return true only when stdout is a real TTY AND we are not in CI. The TUI
uses this to decide between live rendering and plain `@info` lines.

Detection is intentionally conservative — a single false positive (ANSI in a
log file) is much worse than a single false negative (no live board when
one would have worked). `JULIA_TUI_OFF=1` forces the plain-text path for
explicit overrides (e.g. `JULIA_TUI_OFF=1 julia ... > out.log`).
"""
function is_tty()
  return isinteractive() &&
         isa(stdout, Base.TTY) &&
         isempty(get(ENV, "CI", "")) &&
         isempty(get(ENV, "JULIA_TUI_OFF", ""))
end

# ---------------------------------------------------------------------------
# Minimal ANSI helpers — kept inline because the only consumer is below.
# ---------------------------------------------------------------------------

const _ANSI = Dict{Symbol,String}(
  :reset   => "\x1b[0m",
  :bold    => "\x1b[1m",
  :dim     => "\x1b[2m",
  :gray    => "\x1b[90m",
  :cyan    => "\x1b[36m",
  :green   => "\x1b[32m",
  :red     => "\x1b[31m",
  :yellow  => "\x1b[33m",
  :magenta => "\x1b[35m",
)

ansi(c::Symbol) = get(_ANSI, c, "")
ansi(c::Symbol, s::AbstractString) = string(ansi(c), s, ansi(:reset))

# ---------------------------------------------------------------------------
# GPUBoard — static per-GPU "cashier" board
# ---------------------------------------------------------------------------

"""
    GPUBoard(device_names::Vector{String}; max_iter::Int=10000)

Persistent, static board: one row per GPU. Each slot carries a `variant` name
(empty = idle), `iter`/`max_iter` for the progress bar, and `loss`.

On a TTY, every call to `update!` overwrites the previous board in place. In
plain (non-TTY) mode, `update!` only emits `@info` lines when the variant name
changes — finished model and its first iteration both produce one line, not
one per LOG_INTERVAL tick.

Thread-safe: `update!` takes the board's lock. Multiple worker tasks may
call it concurrently; render output is serialized.
"""
mutable struct GPUBoard
  devices::Vector{String}
  variants::Vector{String}
  iter::Vector{Int}
  max_iter::Vector{Int}
  loss::Vector{Float32}
  tty::Bool
  lock::ReentrantLock
  first_render::Bool
end

function GPUBoard(device_names::Vector{String}; max_iter::Int=10000)
  n = length(device_names)
  board = GPUBoard(
    copy(device_names),
    fill("", n),
    fill(0, n),
    fill(max_iter, n),
    fill(Float32(NaN), n),
    is_tty(),
    ReentrantLock(),
    true,
  )
  render!(board, 0, false)  # initial render, no slot change
  return board
end

"""
    update!(board::GPUBoard, idx; variant="", iter=0, max_iter=0, loss=NaN)

Update one device slot. Empty/default keyword arguments leave the existing
field alone (partial update), so callers can tick just iter+loss between
full variant loads without re-stating the variant name.
"""
function update!(board::GPUBoard, idx::Integer;
                 variant::AbstractString="",
                 iter::Integer=0,
                 max_iter::Integer=0,
                 loss::Real=NaN)
  lock(board.lock) do
    variant_changed = false
    if !isempty(variant)
      new_var = String(variant)
      if board.variants[idx] != new_var
        variant_changed = true
      end
      board.variants[idx] = new_var
    end
    if iter > 0;          board.iter[idx]     = Int(iter);       end
    if max_iter > 0;      board.max_iter[idx] = Int(max_iter);   end
    if !isnan(loss);      board.loss[idx]     = Float32(loss);   end
    render!(board, Int(idx), variant_changed)
  end
end

function render!(board::GPUBoard, idx::Integer, variant_changed::Bool)
  n = length(board.devices)
  if board.tty
    if !board.first_render
      # Walk the cursor up over the previous board + footer to overwrite.
      # The board is n+2 lines tall: header + n rows + footer.
      print("\x1b[", n + 2, "A\x1b[J")
    end
    board.first_render = false
    println(ansi(:bold, "── GPU BOARD ──"), " ", n, " device(s)")
    for i in 1:n
      dev = board.devices[i]
      var = board.variants[i]
      it  = board.iter[i]
      mx  = board.max_iter[i]
      ls  = board.loss[i]
      marker_color = isempty(var) ? :gray : :cyan
      marker       = isempty(var) ? "·" : "▸"
      pct  = mx > 0 ? round(100 * it / mx; digits=1) : 0.0
      bar  = build_bar(it, mx)
      loss_str = isnan(ls) ? "" : "  loss $(round(ls; digits=4))"
      println(ansi(marker_color, marker), " ",
              rpad(dev, 24), "  ",
              ansi(marker_color, rpad(isempty(var) ? "idle" : var, 24)), "  ",
              bar, "  ",
              lpad(string(pct, "%"), 6), "  iter ",
              lpad(string(it), 6), "/", lpad(string(mx), 6),
              ansi(:gray, loss_str))
    end
    println(ansi(:gray, "─"^78))
  else
    if board.first_render
      @info "[GPUBoard] initialized $(n) device(s)" devices=board.devices
      board.first_render = false
    end
    # Only emit one log per slot when the variant name changes. Per-tick
    # updates are silent in non-TTY mode (would otherwise drown CI logs).
    if variant_changed && idx > 0
      i = idx
      dev = board.devices[i]
      var = board.variants[i]
      it  = board.iter[i]
      mx  = board.max_iter[i]
      ls  = board.loss[i]
      loss_str = isnan(ls) ? "" : " loss=$(round(ls; digits=4))"
      @info "[GPUBoard] $dev variant=$var iter=$it/$mx$loss_str"
    end
  end
end

# ---------------------------------------------------------------------------
# Bar builder
# ---------------------------------------------------------------------------

function build_bar(iter::Integer, max_iter::Integer; width::Integer=20)
  max_iter <= 0 && return "[" * " "^width * "]"
  filled = clamp(round(Int, width * iter / max_iter), 0, width)
  return "[" * "█"^filled * " "^(width - filled) * "]"
end

# ---------------------------------------------------------------------------
# GridView — compact colored grid for hyperparameter search
# ---------------------------------------------------------------------------

const CELL_PENDING = 0
const CELL_RUNNING = 1
const CELL_DONE    = 2
const CELL_FAILED  = 3

const CELL_GLYPH = Dict(
  CELL_PENDING => "·",
  CELL_RUNNING => "█",
  CELL_DONE    => "▓",
  CELL_FAILED  => "✗",
)

const CELL_COLOR = Dict(
  CELL_PENDING => :gray,
  CELL_RUNNING => :cyan,
  CELL_DONE    => :green,
  CELL_FAILED  => :red,
)

"""
    GridView(rows::Int, cols::Int)

Compact grid-search view: one cell per `(weight1, weight2)` configuration.

States: `pending` (·), `running` (█), `done` (▓), `failed` (✗). On a TTY the
view is re-rendered in place on every transition; in plain mode it only emits
a counter line on first render and on every done/failed transition.
"""
mutable struct GridView
  rows::Int
  cols::Int
  state::Matrix{Int}
  results::Matrix{Float64}
  total::Int
  completed::Threads.Atomic{Int}
  tty::Bool
  lock::ReentrantLock
  first_render::Bool
end

function GridView(rows::Integer, cols::Integer)
  state = fill(CELL_PENDING, rows, cols)
  results = fill(Inf, rows, cols)
  v = GridView(Int(rows), Int(cols), state, results,
               Int(rows * cols), Threads.Atomic{Int}(0), is_tty(),
               ReentrantLock(), true)
  render!(v, false)
  return v
end

"""
    mark_running!(view, i, j)

Transition cell (i,j) to running.
"""
function mark_running!(v::GridView, i::Integer, j::Integer)
  lock(v.lock) do
    v.state[i, j] = CELL_RUNNING
    render!(v, false)
  end
end

"""
    mark_done!(view, i, j, value)

Transition cell (i,j) to done and record its objective.
"""
function mark_done!(v::GridView, i::Integer, j::Integer, value::Real)
  lock(v.lock) do
    v.state[i, j] = CELL_DONE
    v.results[i, j] = Float64(value)
    Threads.atomic_add!(v.completed, 1)
    render!(v, true)
  end
end

"""
    mark_failed!(view, i, j)

Transition cell (i,j) to failed (caught exception during training).
"""
function mark_failed!(v::GridView, i::Integer, j::Integer)
  lock(v.lock) do
    v.state[i, j] = CELL_FAILED
    render!(v, true)
  end
end

function render!(v::GridView, state_changed::Bool)
  rows, cols = v.rows, v.cols
  if v.tty
    if !v.first_render
      # rows + 2 = header + grid + footer
      print("\x1b[", rows + 2, "A\x1b[J")
    end
    v.first_render = false
    pct = round(100 * v.completed[] / v.total; digits=1)
    println(ansi(:bold, "── GRID $(rows)×$(cols) ──"),
            "  ", v.completed[], "/", v.total, " done  (", pct, "%)")
    for i in 1:rows
      print("│ ")
      for j in 1:cols
        print(ansi(CELL_COLOR[v.state[i, j]], CELL_GLYPH[v.state[i, j]]))
      end
      println(ansi(:reset, ""))
    end
    println(ansi(:gray, "─"^78))
  else
    if v.first_render
      @info "[GridView] initialized $(rows)×$(cols)" total=v.total
      v.first_render = false
    elseif state_changed
      @info "[GridView] $(v.completed[])/$(v.total) done"
    end
  end
end

end # module TUI