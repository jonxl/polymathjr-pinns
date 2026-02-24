# Plan: Consolidate Grid Search Output into Single JSON

## Context

The 2D grid search creates a directory per weight configuration (`s0.3-b1.0-p0.7/`), each containing `loss.csv` and `results.json`. A 100×100 grid = 10,000 directories + 20,000 files. This is painful to browse, slow to `ls`, and hard for downstream tools (nn-viewer) to consume.

**Goal:** Replace the entire directory tree with a single `grid_results.json` file. Visualization (surface plots, loss curves) moves to nn-viewer — the grid search code becomes a pure data producer.

@architect: Good grasp of the problem at hand.

@claude-opus-4.6: Thanks. The key constraint is making the grid search a pure data producer — nn-viewer owns everything visual.

## Output

```
data/grid_search/
├── grid_results.json        # all configs + milestones (~30MB for 100×100 grid)
└── top_configs/             # loss.csv for top 10 only
    ├── rank01_s0.3-b1.0-p0.7.csv
    ├── rank02_s0.4-b1.0-p0.6.csv
    └── ...
```

### JSON Schema

```json
{
  "metadata": {
    "timestamp": "2026-02-08T12:34:56",
    "grid_size": 100,
    "neuron_count": 100,
    "maxiters": 10000,
    "optimizer": "adam",
    "seed": 1234,
    "N": 10,
    "num_supervised": 10,
    "domain": [0.0, 1.0],
    "weight1": {"name": "pde", "range": [0.1, 1.0]},
    "weight2": {"name": "supervised", "range": [0.1, 1.0]},
    "fixed_weights": {"bc": 1.0},
    "ode_matrix": [1, 6, 2],
    "benchmark_coefficients": [3.0, -5.0, 11.0, ...]
  },
  "grid": {
    "weight1_values": [0.1, 0.109, ...],
    "weight2_values": [0.1, 0.109, ...],
    "objective_matrix": [[0.234, 0.198, ...], ...]
  },
  "configs": [
    {
      "idx": 0, "i": 0, "j": 0,
      "weights": {"supervised": 0.1, "bc": 1.0, "pde": 0.1},
      "objective": 0.234,
      "coefficients": [1.002, 0.498, ...],
      "milestones": [
        {"iteration": 1000, "objective": 0.891, "coefficients": [0.8, 0.3, ...]},
        {"iteration": 2000, "objective": 0.567, "coefficients": [0.9, 0.4, ...]},
        ...
        {"iteration": 10000, "objective": 0.234, "coefficients": [1.002, 0.498, ...]}
      ]
    },
    {
      "idx": 1, "i": 0, "j": 1,
      "weights": {"supervised": 0.1, "bc": 1.0, "pde": 0.2},
      "objective": 0.312,
      "coefficients": [0.998, 0.501, ...],
      "milestones": [...]
    }
  ]
}
```

@architect: I like this JSON structure with the metadata. Include that with how scaling adam works because we really do not need to repeat the benchmark coeffs or the training coeffs

@claude-opus-4.6: Agreed — `scaling_adam` currently calls `evaluate_solution()` per milestone which calls `append_to_results_json()`, repeating `benchmark_coefficients` and `alpha_matrix` in every single entry. That's the same waste. The fix: `scaling_adam` should produce a similar consolidated JSON with metadata (ode_matrix, benchmark_coefficients written once) and a `milestones` array. That's a separate plan item though — for this plan we're focused on grid search output. I'll note it as follow-up work.

Every config gets `milestones` (objective + coefficients at each milestone interval). Top 10 configs also get their full `loss.csv` written to `top_configs/`. No loss curves embedded in the JSON.


@architect: A couple of things. I think we should still keep the loss.csv files for those top ten then isntead of all 10,000. What is NOT negotiable is being able to plot the objective values from ALL training runs done (so that the nn-viewer that displays it can be able to do so). Another thing to we also want to be able to show the affects of htei ncreasing iterations of some optimization algo. We handled this through a milestone process. We should do the same with the output as well for the grid search. Track the milestones for the outputs from training (same as scaling adam)

@claude-opus-4.6: Three things to address here. Let me go through each:

**1) Top 10 loss.csv files instead of embedded JSON:** Got it. We write `loss.csv` only for the top 10 configs into a `top_configs/` subdirectory. The loss curves stay out of `grid_results.json`. Output becomes:

```
data/grid_search/
├── grid_results.json
└── top_configs/
    ├── rank01_s0.3-b1.0-p0.7.csv
    ├── rank02_s0.4-b1.0-p0.6.csv
    └── ...
```

**2) Objective values for ALL configs must be plottable:** Already covered. Every config in `grid_results.json` has its `objective` value, AND the `grid.objective_matrix` gives nn-viewer the full NxN matrix directly — ready for heatmap/surface rendering. No data loss here.

**3) Milestones for grid search — this is the big one.** Currently grid search trains each config for 10K iterations and only records the *final* objective. You want to see how each config's solution evolves over training — same as `scaling_adam` does with `milestone_interval`.

Here's how this works: `evaluate_weight_configuration()` passes `milestone_interval` and an `on_milestone` callback to `train_pinn()`. At each milestone (say every 1000 iterations), the callback runs `evaluate_solution()` and records `(iteration, objective, coefficients)`. The final `grid_results.json` then stores milestones per config:

```json
{
  "idx": 0,
  "weights": {"supervised": 0.1, "bc": 1.0, "pde": 0.1},
  "objective": 0.234,
  "coefficients": [1.002, 0.498, ...],
  "milestones": [
    {"iteration": 1000, "objective": 0.891, "coefficients": [0.8, 0.3, ...]},
    {"iteration": 2000, "objective": 0.567, "coefficients": [0.9, 0.4, ...]},
    ...
    {"iteration": 10000, "objective": 0.234, "coefficients": [1.002, 0.498, ...]}
  ]
}
```

This lets nn-viewer animate/scrub through training progress for any config — you can see how coefficients converge over iterations across the entire weight landscape. The milestone data is small: ~10 milestones × ~20 coefficients = ~200 numbers per config. For 10K configs that's 2M numbers, maybe 20MB extra in the JSON. Very manageable.

The milestone interval should be configurable in `grid_search_2d()` (default 1000 for 10K iters = 10 milestones per config).

## Files to Modify

| File | Change |
|------|--------|
| `modelcode/PINN.jl` | `train_pinn()`: add `write_loss_csv` kwarg, return `history` as 5th element. `evaluate_solution()`: add `write_results_json` kwarg, return results dict as 2nd element. |
| `utils/two_d_grid_search_hyperparameters.jl` | Rewrite `evaluate_weight_configuration()` to skip file I/O and return data. Rewrite `grid_search_2d()` to collect data and call new `write_grid_results_json()`. Remove `visualize_search_results()`, `save_search_summary()`, and `using Plots`. |
| `src/main.jl` | Update destructuring of `train_pinn()` and `evaluate_solution()` return values. |
| `utils/training_schemes.jl` | Update destructuring at all 4 call sites. |

## Implementation Steps

### Step 1: `train_pinn()` — return history, optional CSV write

**File:** `modelcode/PINN.jl`, function at line 258

- Add kwarg `write_loss_csv::Bool=true`
- Wrap the CSV write block (lines 346-353) in `if write_loss_csv`
- Change return to `return p_trained, coeff_net, st, run_id, history`

### Step 2: `evaluate_solution()` — return results, optional JSON write

**File:** `modelcode/PINN.jl`, function at line 366

- Add kwarg `write_results_json::Bool=true`
- Wrap the `append_to_results_json` call (line 393) and its surrounding `@info` (line 395) in `if write_results_json`
- Collect results dicts into a vector, return `(loss, all_results)`

### Step 3: Update all callers of `train_pinn()` and `evaluate_solution()`

**Files:** `src/main.jl`, `utils/training_schemes.jl`

At each call site, update destructuring to handle new return values:
- `train_pinn`: `p_trained, coeff_net, st, run_id, _ = train_pinn(...)`
- `evaluate_solution`: `function_error, _ = evaluate_solution(...)`

Call sites:
- `src/main.jl:188-189` — SINGLE_RUN
- `training_schemes.jl:44-45` — `scaling_neurons`
- `training_schemes.jl:89` — `scaling_adam` (on_milestone callback)
- `training_schemes.jl:94` — `scaling_adam` train_pinn call
- `training_schemes.jl:117-118` — `scaling_lbfgs`

### Step 4: Rewrite `evaluate_weight_configuration()` — add milestones

**File:** `utils/two_d_grid_search_hyperparameters.jl`, lines 38-87

- Remove `config_dir` creation (`mkpath`) and `data_directories` array
- Add `milestone_interval` parameter (default 1000)
- Define `on_milestone` callback that calls `evaluate_solution(...; write_results_json=false)` and collects `(iteration, objective, coefficients)` into a milestones vector
- Call `train_pinn(settings, base_data_dir; write_loss_csv=false, progress_callback=..., milestone_interval=milestone_interval, on_milestone=on_milestone)`
- Call `evaluate_solution(...; write_results_json=false)` for final result
- Return `(total_error, loss_history, eval_results, milestones)` instead of just `total_error`

### Step 5: Rewrite data collection in `grid_search_2d()`

**File:** `utils/two_d_grid_search_hyperparameters.jl`, lines 115-226

- Add `using JSON, Dates, DataFrames, CSV` to module imports
- Add `milestone_interval` kwarg to `grid_search_2d()` (default 1000)
- Pre-allocate `all_loss_histories`, `all_config_results`, and `all_milestones` vectors (length = total_configs)
- Update spawned tasks to return `(j, i, objective_value, loss_history, eval_results, weights, milestones)`
- After each batch, store results in the pre-allocated vectors using unique linear indices
- After all batches: call `write_grid_results_json(...)` and `write_top_loss_csvs(...)` instead of `visualize_search_results()` + `save_search_summary()`

### Step 6: Implement `write_grid_results_json()`

**File:** `utils/two_d_grid_search_hyperparameters.jl` (new function)

- Build the JSON structure (metadata, grid, configs) from collected data
- Every config gets: `idx`, `i`, `j`, `weights`, `objective`, `coefficients`, `milestones`
- `milestones` is an array of `{"iteration": N, "objective": X, "coefficients": [...]}`
- Extract `ode_matrix` and `benchmark_coefficients` from training data / eval results for metadata
- Write with `JSON.print(io, output, 2)` to `{base_data_dir}/grid_results.json`

### Step 7: Implement `write_top_loss_csvs()`

**File:** `utils/two_d_grid_search_hyperparameters.jl` (new function)

- Sort configs by objective, take top 10
- For each, write `{base_data_dir}/top_configs/rankNN_sX-bY-pZ.csv` with columns: `iteration, total, bc, pde, supervised`
- Uses the loss history returned from `train_pinn()`

### Step 8: Remove dead code

**File:** `utils/two_d_grid_search_hyperparameters.jl`

- Delete `visualize_search_results()` (lines 346-433)
- Delete `save_search_summary()` (lines 483-508)
- Remove `using Plots` (line 3) — no longer needed in this module
- Remove the calls to these functions in `grid_search_2d()`

## Follow-up Work (not in this plan)

- Apply same consolidated JSON output pattern to `scaling_adam` — metadata with benchmark_coefficients/ode_matrix written once, milestones array per training example

## Thread Safety

| Concern | Status | Why |
|---------|--------|-----|
| `all_loss_histories[k]` writes | Safe | Each `k` unique per task |
| `all_config_results[k]` writes | Safe | Same |
| `objective_matrix[j,i]` writes | Safe | Unchanged from before |
| JSON file write | Safe | Single write after all tasks complete |
| No per-config I/O during training | Safe | `write_loss_csv=false`, `write_results_json=false` |

## What Does NOT Change

- `loss_fn()` / loss functions — untouched
- `gpu_utils.jl` — untouched
- `helper_funcs.jl` — `append_to_results_json` stays for SINGLE_RUN / SCALING_ADAM modes
- `ProgressBar.jl` — untouched
- SINGLE_RUN and SCALING_ADAM training modes — unchanged behavior (default kwargs preserve old file writes)
- `GridSearchResult` struct — unchanged (still returned from `grid_search_2d`)
- `random_search_2d()` — untouched (separate code path)

## Verification

1. Run a small grid search (2×2 grid): `julia --project -t auto src/main.jl` with `TRAINING_MODE = "GRID_SEARCH"` and `num_points = 2`
2. Verify `data/grid_search/grid_results.json` exists and contains:
   - `metadata` with correct search params, ode_matrix, benchmark_coefficients
   - `grid.objective_matrix` as 2×2 array
   - 4 config entries, each with `milestones` array
   - Each milestone has `iteration`, `objective`, `coefficients`
3. Verify `data/grid_search/top_configs/` contains loss.csv files for top configs (up to 4 for a 2×2 grid)
4. Verify NO per-config directories (`s0.1-b1.0-p0.1/` etc.) were created
5. Verify SINGLE_RUN mode still writes `results/loss.csv` and `results/results.json` as before
6. Validate JSON with `python3 -c "import json; json.load(open('data/grid_search/grid_results.json'))"`
