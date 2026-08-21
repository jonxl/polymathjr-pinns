# generalization_variants.jl — Generalization experiments × representations

The five diagnostic shapes from `utils/experiments.jl` (transfer, extrapolate,
cross-region-range, showcase, gen_radius) each test a different facet of
generalization. They were originally run sequentially per rep
(`scripts/shared/run_all.jl`); this module threads them through the same
per-GPU variant runner that `staged_variants.jl` uses for single-ODE training.

**Location:** `utils/generalization_variants.jl` (entry point:
`scripts/generalization_variants.jl`)

---

## Mental model

```
rep × experiment   →   GeneralizationSpec
                       ↓
                     GenJob (spec + device_id + output_dir)
                       ↓
                     run_generalization_variants (one task per job)
                       ↓
                     GPUBoard (one slot per visible device)
```

```
── GPU BOARD ── 5 device(s)
│ ▸ GPU 0 (A100)      transfer_power_series    ...
│ ▸ GPU 1 (A100)      transfer_eigenvalue      ...
│ ▸ GPU 2 (A100)      extrapolate_power_series ...
│ ▸ GPU 3 (A100)      extrapolate_eigenvalue   ...
│ ▸ GPU 4 (A100)      range_power_series       ...
```

Each worker task binds `CUDA.device!(i)` per the round-robin assignment and
dispatches on `spec.name` to the matching `Experiments.run_*` function. The
two representations share no mutable state, so they can run truly concurrently.

---

## API

### `GeneralizationSpec`

```julia
GeneralizationSpec(name, representation;
                   regions=REGIONS, Rmax=6, Ng=31,
                   n_per_region=200, maxiters=3000)
```

`name` is the diagnostic shape: `:transfer`, `:extrapolate`, `:range`
(cross-region-range), `:showcase`, or `:gen_radius`.

Unused fields default silently — `:showcase` ignores `regions`/`Rmax`/`Ng`
entirely, while `:transfer` ignores `Rmax`/`Ng`.

### `default_generalization_specs`

```julia
default_generalization_specs(;
    reps=(:power_series, :eigenvalue),
    experiments=(:transfer, :extrapolate, :range, :showcase, :gen_radius),
    kwargs...)
```

Cross-product of reps × experiments. The user can override either axis to
narrow the run; `kwargs...` are forwarded to `GeneralizationSpec`.

### `build_gen_jobs`

```julia
build_gen_jobs(specs, output_root; n_devices=0) → Vector{GenJob}
```

Resolve a list of `GeneralizationSpec`s into `GenJob`s with round-robin
device assignment over the visible CUDA devices. Identical convention to
`Variants.build_jobs` — the same set of specs lands on the same slots across
runs.

### `run_generalization_variants`

```julia
run_generalization_variants(jobs::Vector{GenJob})
```

Spawn one worker task per job. Each worker:

1. Binds `CUDA.device!(job.device_id - 1)` as its **first** CUDA action
   (per `plans/active/staged-gpu-runs.md` II.1 rule 1).
2. Builds an `ExperimentConfig` for the spec's representation.
3. Dispatches on `spec.name` to the matching `Experiments.run_*` function.
4. Saves the resulting PanelSet to `job.output_dir/panelset.json`.
5. Updates the GPUBoard slot for its device.

The function blocks until all jobs finish.

---

## Entry point: `scripts/generalization_variants.jl`

```bash
julia --project=. -t auto scripts/generalization_variants.jl \
    --reps power_series,eigenvalue \
    --experiments transfer,extrapolate,range,showcase,gen_radius \
    --Rmax 6 --Ng 31 \
    --n-per-region 200 --maxiters 3000 \
    --output-root results/gen-$(date +%m-%d-%y-%H%M)
```

The default spec list runs all 5 experiments × both reps = 10 jobs. On a
10-GPU box each job gets its own device; on a smaller box they queue
round-robin.

Output structure:

```
results/gen-{timestamp}/
├── gen-transfer-power_series/
│   ├── panelset.json                 ← PanelSet (loadable by viz viewer)
│   └── exp-transfer_power_series-train_saddle/
│       ├── model.checkpoint
│       └── loss.csv
├── gen-transfer-eigenvalue/
│   └── ...
```

---

*See also: [TUI](tui.md), [Variants](variants.md), [Experiments.jl](experiments.md)*