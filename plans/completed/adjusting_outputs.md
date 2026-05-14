# Plan: Adaptive Run-ID Output System

## Context

Currently every training run overwrites `results.json` (single dict) and writes to a hardcoded `iteration_output.csv`. When running scaling experiments (e.g. `scaling_adam` with 3 iteration counts), each run clobbers the previous result. The goal is to make output adaptive: each run gets a unique ID like `adam-a7x9k2m1`, CSVs are named by that ID, and `results.json` accumulates all runs as an array.

## Changes

### 1. `utils/helper_funcs.jl` — Add two helpers

- Add `using Random` and `using JSON` imports
- Add `generate_run_id(optimizer::String)::String` — creates `"{optimizer}-{randstring(8)}"`
- Add `append_to_results_json(results_file, run_id, results_dict)` — reads existing array (or wraps old single-dict format), pushes new entry with `"id"` field, writes back
- Export both new functions

### 2. `modelcode/PINN.jl` — Core signature changes

**PINNSettings** (line 62): Add `optimizer::String` field at end of struct

**train_pinn** (line 257): Change signature from `(settings, csv_file)` to `(settings, output_dir)`:
- Generate `run_id = generate_run_id(settings.optimizer)` at top
- Derive `csv_file = joinpath(output_dir, "$(run_id).csv")`
- Return `(p_trained, coeff_net, st, run_id)` instead of `(p_trained, coeff_net, st)`

**evaluate_solution** (line 347): Change signature from `(settings, p_trained, coeff_net, st, benchmark_dataset, results_file)` to `(settings, p_trained, coeff_net, st, benchmark_dataset, output_dir, run_id)`:
- Replace `open(results_file, "w") ... JSON.print` with `append_to_results_json(joinpath(output_dir, "results.json"), run_id, results)`

### 3. `src/main.jl` — Update single-run caller

- Remove `csv_file` and `results_file` path construction (lines 186-187)
- Add `"adam"` as last arg to PINNSettings constructor (line 189)
- Update `train_pinn` call to pass `output_dir`, destructure 4th return value `run_id`
- Update `evaluate_solution` call to pass `output_dir, run_id`

### 4. `utils/training_schemes.jl` — Update scaling callers

For `scaling_neurons`, `scaling_adam`, `scaling_lbfgs`:
- Remove `csv_file` and `results_file` path construction
- Add optimizer string (`"adam"` or `"lbfgs"`) as last PINNSettings arg
- Update `train_pinn` call: pass `output_dir`, capture `run_id`
- Update `evaluate_solution` call: pass `output_dir, run_id`

### 5. `utils/two_d_grid_search_hyperparameters.jl` — Update grid search

In `evaluate_weight_configuration`:
- Add `"adam"` to PINNSettings constructor (line 70)
- Update `train_pinn` call: pass `config_dir`, capture `run_id`
- Update `evaluate_solution` call: pass `config_dir, run_id`

## Result Format

**Before** — `results/results.json`:
```json
{"alpha_matrix": [...], "benchmark_coefficients": [...], "pinn_coefficients": [...], "function_error": 0.123}
```

**After** — `results/results.json`:
```json
[
  {
    "id": "adam-a7x9k2m1",
    "alpha_matrix": [...],
    "benchmark_coefficients": [...],
    "pinn_coefficients": [...],
    "function_error": 0.123
  },
  {
    "id": "adam-b3c8d2e5",
    "alpha_matrix": [...],
    ...
  }
]
```

**CSV files**: `results/adam-a7x9k2m1.csv`, `results/adam-b3c8d2e5.csv`, etc.

## Files Modified (5)

| File | What changes |
|------|-------------|
| `utils/helper_funcs.jl` | +`generate_run_id`, +`append_to_results_json`, +imports |
| `modelcode/PINN.jl` | PINNSettings +`optimizer` field, train_pinn/evaluate_solution signatures |
| `src/main.jl` | Update caller to new signatures |
| `utils/training_schemes.jl` | Update 3 scaling callers to new signatures |
| `utils/two_d_grid_search_hyperparameters.jl` | Update grid search caller to new signatures |

---

## Phase 2: Milestone-Based Training Evaluation

### Context

With Phase 1 complete, `scaling_adam()` still restarts training from scratch for each iteration count (1000, 10000, 100000). This wastes compute. The fix: train ONCE up to `max(milestones)` and evaluate mid-training at each milestone via a callback.

### Changes

**`src/main.jl`** — Replace `iteration_counts` Dict with `milestones = [1000, 10000, 100000]` vector. Pass to `scaling_adam()`.

**`utils/training_schemes.jl`** — `scaling_adam(settings, milestones::Vector{Int})`:
- Single loop over training_dataset (no outer loop over iteration counts)
- Creates PINNSettings with `maxiters = maximum(milestones)`
- Defines `on_milestone` closure that calls `evaluate_solution(...; iteration=iteration)`
- Passes `milestones` + `on_milestone` to `train_pinn()`
- Removes `benchmark_losses.csv` writing

**`modelcode/PINN.jl`** — `train_pinn()`:
- New kwargs: `milestones::Vector{Int}`, `on_milestone::Union{Function,Nothing}`
- Converts milestones to `Set` for O(1) lookup
- In `custom_callback`: checks `length(history) in milestone_set`, calls `on_milestone(p_current, iteration, coeff_net, st, run_id)`
- CSV path changed from `"$(run_id).csv"` to `"loss.csv"`

**`modelcode/PINN.jl`** — `evaluate_solution()`:
- New kwarg: `iteration::Int=0`
- Includes `"iteration" => iteration` in results Dict

### Output

- `results/results.json` — one entry per milestone with `"iteration"` field
- `results/loss.csv` — single file with full loss history
