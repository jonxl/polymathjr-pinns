# Experiment presets

The shared package is `PINNStudies` (`src/PINNStudies.jl`). Active training and
evaluation entry points import the same modules. Direct includes and the old
`power_series_variation` name remain available for legacy scripts and results.
New model metadata uses `power_series` plus `coefficient_weighting = "descending"`.

Available presets:

- `power_series_standard.toml`: power series with uniform coefficient supervision.
- `power_series_descending.toml`: the same representation with descending coefficient weights.
- `eigenvalue_standard.toml`: eigenvalue representation with solution-space supervision.

Each preset has independent model, objective, data, training, and run sections.
Model and data seeds are separate. A data count of zero selects the existing
experiment-specific count. Geometry comparisons use trace/determinant inputs for
both representations. Explicit command-line options override preset values.

Preview one preset without creating a run:

```bash
julia --project=. scripts/generalization_variants.jl \
  --config configs/power_series_descending.toml --dry-run
```

Run one preset and export its report:

```bash
julia --project=. -t auto scripts/generalization_variants.jl \
  --config configs/power_series_descending.toml --report
```

Compare the three presets with the existing million-iteration study budget:

```bash
nohup bash scripts/shared/run_generalization_study.sh > launch.log 2>&1 < /dev/null &
```

To use a preset through Bash, set `STUDY_CONFIG`. Its iteration budget is used
unless an explicit third positional argument overrides it. Without a preset,
the generalization launcher retains its 1,000,000-iteration default.
`--variants power_series_standard,power_series_descending,eigenvalue_standard`
selects comparison variants; the old `--reps` flag and names are accepted too.

Each new run gets `results/YYYY-MM-DD/run-TIME-ID/` containing:

- `config.toml`: resolved settings, including each experiment job's defaults.
- `manifest.json`: source hashes, job status/errors, and report status.
- `source/`: a snapshot of the code and dependency manifests used for the run.
- `run.log`: warnings, errors, and concise status messages, without terminal redraws.
- `jobs/`: model checkpoints, per-model configurations, status, and loss histories.
- `report/`: HTML and exported figures when `--report` is requested.

`launch.log` prints the run location. An explicit `--output-root` must be empty.
The portable CUDA study also uses date/run directories by default; an explicit
`DUAL_STUDY_ROOT` retains its existing immutable-bundle validation and completed-model reuse.
It keeps its existing seed-specific subdirectory inside the run directory.

The optimizer remains Adam with the existing learning rate and objective:
`pde_weight * num_supervised * PDE + supervised_weight * supervision`.
Initial-condition loss is diagnostic. Changing N also changes `num_supervised`,
so objective magnitudes across N are not directly comparable. New CSVs store
weighted contributions; new plots separate those from raw diagnostics. Old
histories are displayed without inventing missing weights.

Solution accuracy is evaluated against the analytic solution. Power-series
reports additionally record true-coefficient truncation error at the same
points. These are different errors, not additive quantities; larger N does not
guarantee better learned accuracy. Degree sweeps require coefficient targets
through the largest requested N. Saved approximation summaries identify their
evaluation domain (currently training equations); held-out comparisons remain
in the experiment's evaluation panels.

Interrupted training returns an explicit status and is not saved as a completed
model. Independent failed jobs are recorded while other jobs finish; the launcher
then exits unsuccessfully. A hard system kill cannot update status, so a manifest
left at `running` is not proof that a process is alive. Model checkpoints support
weight restoration; exact optimizer/RNG-state resume is not introduced here.

This refactor does not resolve the previous full-study out-of-memory failure.
