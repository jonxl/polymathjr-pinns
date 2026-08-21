# tui.jl — Terminal UI helpers

Live progress displays for the variant runner and grid search. Both renderers
detect whether stdout is a real TTY and fall back to plain `@info` lines when
it is not, so CI logs and file-redirected runs stay clean.

**Location:** `utils/tui.jl`

---

## `is_tty() → Bool`

Return `true` only when:

1. `isinteractive()` — a session is attached,
2. `stdout` is a `Base.TTY`,
3. `CI` and `JULIA_TUI_OFF` env vars are both empty.

Detection is intentionally conservative: a single false positive (ANSI escape
codes in a log file) is much worse than a single false negative (no live
board when one would have worked).

Set `JULIA_TUI_OFF=1` to force the plain-text path explicitly, e.g.

```bash
JULIA_TUI_OFF=1 julia --project=. scripts/staged_variants.jl > out.log
```

---

## `GPUBoard` — static per-gPU "cashier" board

A persistent, static board: one row per GPU. Each slot carries a `variant`
name (empty = idle), `iter`/`max_iter` for the progress bar, and `loss`.

On a TTY, every call to `update!` overwrites the previous board in place.
In plain mode, `update!` only emits `@info` lines when the variant name
changes — a finished model and its first iteration both produce one line,
not one per `LOG_INTERVAL` tick.

```
┌── GPU BOARD ── 5 device(s)
│ ▸ GPU 0 (A100)      ps_N20    [████████████████████]  66.7%  iter 2000/3000  loss 0.0012
│ ▸ GPU 1 (A100)      ps_N25    [██████████████      ]  50.0%  iter 1500/3000  loss 0.0015
│ ▸ GPU 2 (A100)      ps_N30    [███████████         ]  33.3%  iter 1000/3000  loss 0.0021
│ · GPU 3 (A100)      idle
│ · GPU 4 (A100)      idle
```

### API

```julia
board = TUI.GPUBoard(device_names::Vector{String}; max_iter::Int=10000)

TUI.update!(board, idx; variant="", iter=0, max_iter=0, loss=NaN)
```

`update!` takes partial updates — empty keyword arguments leave the existing
field alone, so a worker can tick just `iter`+`loss` between full variant
loads without re-stating the variant name.

Thread-safe: `update!` takes the board's `ReentrantLock`. Multiple worker
tasks may call it concurrently; render output is serialized.

---

## `GridView` — compact colored grid for hyperparameter search

One cell per `(weight1, weight2)` configuration.

States: `pending` (·), `running` (█), `done` (▓), `failed` (✗). On a TTY
the view is re-rendered in place on every transition; in plain mode it
emits a counter line on first render and on every done/failed transition.

```
── GRID 4×4 ──  7/16 done  (43.8%)
│ ██▓·▓·▓·▓·
│ ██▓·▓·▓·▓·
│ ██▓·▓·▓·▓·
│ ██▓·▓·▓·▓·
```

### API

```julia
view = TUI.GridView(rows::Int, cols::Int)

TUI.mark_running!(view, i, j)
TUI.mark_done!(view, i, j, value)
TUI.mark_failed!(view, i, j)
```

All mutators take the view's `ReentrantLock`. The completed counter is a
`Threads.Atomic{Int}` so workers can increment it without contention.

---

## When to use each

| Module                      | When                                      |
| --------------------------- | ----------------------------------------- |
| `TUI.GPUBoard`              | Multi-GPU variant runs, one model per GPU |
| `TUI.GridView`              | Grid search with visible progress         |
| `ProgressBar.ProgressBar`   | Single-model training (legacy path)       |

`ProgressBar` (from `utils/ProgressBar.jl`) is still used for the default
single-run training; the TUI modules target the multi-model orchestration
paths.

---

*See also: [Variants](variants.md), [PINN.jl](pinn.md)*