# variants.jl — Variant registry + per-GPU dispatch

The variant runner spawns one worker task per variant, binds
`CUDA.device!(i)` per the round-robin assignment, and drives a single
`TUI.GPUBoard` so all devices are visible simultaneously.

**Location:** `utils/variants.jl` (entry point: `scripts/staged_variants.jl`)

---

## Mental model

A variant is a leaf in a tree:

```
representation (:power_series | :eigenvalue)   ← root, fixes loss form
  neuron_count, N, maxiters, seed_offset      ← free parameters
```

The variant name (e.g. `"ps_N20"`, `"eig"`) is the durable identifier —
it goes into checkpoint metadata and output directory names.

```
┌── GPU BOARD ── 5 device(s)
│ ▸ GPU 0 (A100)      ps_N20    ...
│ ▸ GPU 1 (A100)      ps_N25    ...
│ ▸ GPU 2 (A100)      ps_N30    ...
│ ▸ GPU 3 (A100)      ps_N35    ...
│ ▸ GPU 4 (A100)      eig       ...
```

When jobs outnumber devices, later jobs queue round-robin on the same
devices; the GPUBoard then shows the LAST-loaded variant on each device.

---

## API

### `VariantSpec`

```julia
VariantSpec(name, representation;
             N=20, neuron_count=64, maxiters=3000, seed_offset=0)
```

`representation` must be `:power_series` or `:eigenvalue` (anything else
raises). The four inseparable operations of the representation
(input encoding, output width, loss triple, reconstruction of `u(x)`)
move together — see `plans/active/staged-gpu-runs.md` III.1.

### `build_jobs`

```julia
build_jobs(specs::Vector{VariantSpec}, dataset::Dict, output_root::String;
           n_devices::Int=0) → Vector{VariantJob}
```

Resolve a list of `VariantSpec`s into `VariantJob`s with round-robin device
assignment over the visible CUDA devices (or one CPU slot when no GPU).

Round-robin gives a stable assignment (same spec list always lands on the
same slots) and matches the "one model per GPU" pattern (Q2 in
`plans/active/staged-gpu-runs.md`).

### `run_staged_variants`

```julia
run_staged_variants(jobs::Vector{VariantJob};
                    benchmark_dataset::Union{Dict,Nothing}=nothing)
```

Spawn one worker task per job. Each worker:

1. Binds `CUDA.device!(job.device_id - 1)` as its **first** CUDA action
   (per `plans/active/staged-gpu-runs.md` II.1 rule 1).
2. Builds its `PINNSettings` from the job spec + shared dataset.
3. Calls `train_pinn` with a `progress_callback` that updates the
   GPUBoard slot's iter + loss.
4. Saves a checkpoint under `job.output_dir/model.checkpoint`.

The function blocks until all jobs finish.

### `device_names`

```julia
device_names() → Vector{String}
```

`["GPU 0 (A100)", "GPU 1 (A100)", ...]` when CUDA is available;
`["CPU"]` otherwise. Speaks the **visible** device space
(`CUDA_VISIBLE_DEVICES` renumbers physical → visible; we use the visible
space, which is what `CUDA.device!` accepts).

### `generate_shared_dataset`

```julia
generate_shared_dataset(; n_per_region=200, num_terms=21,
                        regions=Plugboard.TRACE_DET_REGIONS, seed=1234) → Dict
```

Generate a single fixed, seeded ODE dataset, shared by every variant in a
run. Deterministic given `seed` — same seed → same dataset.

### `write_manifest`

```julia
write_manifest(path, jobs, dataset_meta; n_devices)
```

Persist a run manifest: job list, device assignments, dataset metadata.
Lets a future restart decide which jobs are done / running / fresh.

---

## Entry point: `scripts/staged_variants.jl`

```bash
julia --project=. -t auto scripts/staged_variants.jl \
    --specs power_series,eigenvalue \
    --neurons 64 \
    --maxiters 3000 \
    --n-per-region 200 \
    --output-root results/staged-$(date +%m-%d-%y-%H%M) \
    --seed 1234
```

The default spec list runs four power-series variants (N=20, 25, 30, 35)
plus the eigenvalue variant — five jobs total. On a five-GPU box each GPU
gets exactly one variant; on a smaller box they queue round-robin.

---

*See also: [TUI](tui.md), [PINN.jl](pinn.md)*