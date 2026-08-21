# Staged GPU Runs + utils/ Reorganization

## Status: SPEC — revision 3. All open items resolved. Nothing implemented yet.

@claude-opus-5: Revision 3 closes the last four open items with @architect's
answers — weights both 1.0, variant named `:eigenvalue`, CUDA_VISIBLE_DEVICES
remapping confirmed, throughput spike deferred. No item blocks implementation.

Carried from revision 2:
- Q1-Q5 resolved (see "Resolved Decisions"). Q1 is **verified from CUDA.jl
  source**, not model knowledge, and the finding is more subtle than assumed.
- Variant model corrected: representation is the **root**, not a peer axis.
- Staging scope fixed at the two representations, with a modularity constraint.
- Part III (variant registry) and II.6 (job-list builder).

Three separable pieces: Part I is a pure-refactor reorg of `utils/`; Part II is
the staged scheduler; Part III is the variant registry. Part I lands first and
independently — see "Sequencing".

---

## Motivation

N models, M GPUs, "cashiers": dispatch one training job per GPU, pull the next
job as each GPU frees, generically across every experiment configuration rather
than hand-rolled per experiment.

The workloads are all sequential loops over independent `train_pinn` calls:

- `run_transfer` — one model per region [source: utils/experiments.jl:L296-L310]
- `run_sweep` — one model per parameter value [source: utils/experiments.jl:L265-L272]
- `run_all.jl` — 6 shapes x 2 representations, ~60 min
  [source: scripts/shared/run_all.jl:L9-L18]
- `grid_search_2d` — up to 10,000 configs
  [source: plans/ideas/batched-pinn-training-idea.md — "100x100 = 10,000"]

`train_pinn` is already a near-perfect unit of work: takes `settings` +
`output_dir`, allocates its own network [source: architectures/PINN.jl:L420],
builds its own buffers [source: architectures/PINN.jl:L428-L431], writes its own
artifacts. It touches no global mutable state except `GPUUtils`.

That one exception is the blocker.

---

## Resolved Decisions (@architect, this session)

| # | Question | Decision |
|---|---|---|
| Q1 | CUDA.jl per-task device + math mode semantics | **VERIFIED FROM SOURCE** — see II.1. Task-local confirmed; a global-default hazard was found. |
| Q2 | k jobs per GPU? | **One model per GPU (k=1)**, but a freed GPU immediately takes the next queued job. This is the work-stealing queue, not a batch barrier. k remains a parameter defaulting to 1. |
| Q3 | CPU fallback? | **No.** Hard error when no GPU. Staged runs are GPU-only. |
| Q4 | Manifest location | `results/staged-{run_id}/manifest.json`, following `generate_run_id` [source: utils/helper_funcs.jl:L137] |
| Q5 | docs/ updates | **Deferred** — single doc pass after all parts land. |

### Staging scope for the first milestone

Compare **exactly two variants**: the power-series representation and the
trace-determinant/eigenvalue representation
[source: architectures/PINN.jl:L85-L87].

**Constraint, stated as a requirement not an aspiration:** this does NOT make
the GPU machinery representation-aware. The device pool, scheduler, and
job-list builder must never name a representation. They take variants as opaque
values. The place this will be tempting to violate is the job-list builder,
where with 2 variants a hardcoded pair is one line shorter than a registry
lookup. Do not.

### Loss weights: out of scope as a staging axis

`supervised_weight` and `pde_weight` [source: architectures/PINN.jl:L78-L79] are
held **fixed** for this milestone; they are not a staged axis.

**RESOLVED (@architect):** both weights are **1.0** for the power series.
`supervised_weight = 1.0f0`, `pde_weight = 1.0f0`.

This is already the `ExperimentConfig` default
[source: utils/experiments.jl:L92-L95], so no config change is required — the
staged milestone runs at the existing defaults, and the weights simply never
enter the axis set.

---

## The core blocker: device is a module-level singleton

```julia
const GPU_AVAILABLE = Ref{Bool}(false)
const DEVICE = Ref{Union{CuDevice, Nothing}}(nothing)

function __init__()
    GPU_AVAILABLE[] = CUDA.functional()
    if GPU_AVAILABLE[]
        DEVICE[] = CUDA.device()
        CUDA.math_mode!(CUDA.PEDANTIC_MATH)
    end
end
```
[source: utils/gpu_utils.jl:L8-L20]

Set once at load. No `device_id` parameter exists anywhere in the module
[source: utils/gpu_utils.jl:L30-L52]. `train_pinn` consumes only a boolean:

```julia
use_gpu = GPUUtils.is_gpu_available()                     # L404
to_device_fn = x -> GPUUtils.to_device(x; gpu=use_gpu)    # L424
```
[source: architectures/PINN.jl:L404, L424]

Existing parallelism is a fixed-size batch barrier — every batch waits on its
slowest member [source: utils/two_d_grid_search_hyperparameters.jl:L176-L200] —
and `estimate_batch_size()` calls `CUDA.available_memory()` with no device
argument, measuring device 0 and assuming homogeneity
[source: utils/two_d_grid_search_hyperparameters.jl:L99-L111].

---

# PART I — utils/ Reorganization

## Current state

12 flat files, ~4,086 lines [source: `wc -l utils/*.jl`, this session]:

| File | Lines |
|---|---|
| loss_functions.jl | 784 |
| experiments.jl | 687 |
| two_d_grid_search_hyperparameters.jl | 624 |
| plugboard.jl | 582 |
| two_d_grid_optimitze_hyperparameters.jl | 451 |
| training_schemes.jl | 251 |
| safetensors_utils.jl | 212 |
| helper_funcs.jl | 176 |
| binary_search_on_weights.jl | 139 |
| snapshot_utils.jl | 96 |
| gpu_utils.jl | 56 |
| ProgressBar.jl | 28 |

## Target layout

```
utils/
  gpu/
    gpu_utils.jl                        (moved)
    device_pool.jl                      (NEW — Part II)
    scheduler.jl                        (NEW — Part II)
  hyperparams/
    two_d_grid_search_hyperparameters.jl    (moved)
    two_d_grid_optimize_hyperparameters.jl  (moved + TYPO FIXED)
    binary_search_on_weights.jl             (moved)
  data/
    plugboard.jl                        (moved)
  training/
    training_schemes.jl                 (moved)
    experiments.jl                      (moved)
    variants.jl                         (NEW — Part III)
  io/
    safetensors_utils.jl                (moved)
    snapshot_utils.jl                   (moved)
  core/
    loss_functions.jl                   (moved)
    helper_funcs.jl                     (moved)
  ProgressBar.jl                        (stays at root)
```

### Rationale per group

- **`gpu/`** — grows from 56 lines to a real subsystem in Part II.
- **`hyperparams/`** — all three are weight-space search; 1,214 lines together.
- **`data/`** — @architect: plugboard is a synthetic data generator, not a math
  primitive. Confirmed by exports: `generate_random_ode_dataset`,
  `generate_region_dataset`, `generate_shell_dataset`, `generate_grid_dataset`,
  `sample_region` [source: utils/plugboard.jl:L580-L581].
- **`training/`** — orchestration; Part III's registry joins it.
- **`io/`** — persistence. `snapshot_utils` already depends on
  `safetensors_utils` [source: utils/snapshot_utils.jl:L7-L8].
- **`core/`** — primitives. `loss_functions` already includes `helper_funcs`
  [source: utils/loss_functions.jl:L6].
- **`ProgressBar.jl`** stays at root — 28 lines, included from `architectures/`
  rather than within utils [source: architectures/PINN.jl:L44,
  architectures/PINN_specific.jl:L41].

### Typo fix

`two_d_grid_optimitze_hyperparameters.jl` -> `..._optimize_...`. Fix during the
move since the path changes anyway. [source: `ls utils/` — reads "optimitze"]

## Risk: 54 include sites, three path conventions

**54 `include("...")` calls** across utils/architectures/src/scripts/viz/test
[source: repo-wide grep, this session]. Three inconsistent styles already exist:

- `include("../utils/safetensors_utils.jl")` — from *inside* utils/, via a
  parent hop [source: utils/snapshot_utils.jl:L7]
- `include("./helper_funcs.jl")` [source: utils/loss_functions.jl:L6]
- `include("plugboard.jl")` — bare sibling [source: utils/experiments.jl:L38]

That `../utils/` self-reference from within `utils/` is evidence paths have
already drifted. Julia's `include` fails at load time, so a missed path is a
hard error — noisy but safe.

### Acceptance criteria for Part I

- [ ] Every module loads cleanly in a fresh session.
- [ ] Existing tests pass unchanged: `test_integration.jl`,
      `test_memorization.jl`, `test_safetensors.jl`,
      `test_semigroup_generalization.jl`, `test_heldout_benchmark.jl`
      [source: `find test -name "*.jl"`]
- [ ] Zero behavioral diff. Paths and one filename only.
- [ ] Include style normalized to one convention within utils/.

**Own commit, before any Part II/III work.** Mixed into new concurrency code, an
include bug and a scheduler bug become indistinguishable.

---

# PART II — Staged GPU Scheduler

## II.1 Device semantics — VERIFIED against the pinned CUDA.jl

**Verified environment**: CUDA.jl at `~/.julia/packages/CUDA/TPbi4` (pinned by
`Manifest.toml`), Julia 1.12.6. Read from source this session. Note: the dev
machine has **no GPU** (`nvidia-smi` fails), so this is source-verified, not
runtime-confirmed.

### Finding 1 — device and math mode ARE task-local (design holds)

`TaskLocalState` holds `device`, `context`, `streams`, `math_mode`,
`math_precision` [source: lib/cudadrv/state.jl:L41-L46], stored per-task in
`task_local_storage()[:CUDA]` [source: lib/cudadrv/state.jl:L69-L81].

- `device!` sets `state.device` / `state.context` on the calling task
  [source: lib/cudadrv/state.jl:L279-L286]
- `math_mode!` sets `state.math_mode` on the calling task
  [source: lib/cudadrv/state.jl:L336-L339]

So one worker per device, each calling `CUDA.device!(i)` once, is sound.

### Finding 2 — HAZARD: both also write process-global defaults

```julia
default_device[]    = dev     # inside device!,    L275
default_math_mode[] = mode    # inside math_mode!, L340
```
[source: lib/cudadrv/state.jl:L275, L340]

New tasks **inherit from those globals** on first CUDA touch:

```julia
function TaskLocalState(dev::CuDevice = something(default_device[], CuDevice(0)),
                        ctx::CuContext = context(dev))
    math_mode = something(default_math_mode[], ...)
```
[source: lib/cudadrv/state.jl:L48-L51]

The source says so explicitly: math mode is *"sticky (once set on a task,
inherit to newly created tasks)"* [source: lib/cudadrv/state.jl:L32], and
`default_device` is *"the default device unitialized tasks will use, set when
switching devices"* [source: lib/cudadrv/state.jl:L36-L38].

**Consequence:** `device!` is not purely local. With M workers binding at
startup, `default_device[]` ends up holding whichever won the race. Benign **iff
every task binds explicitly before touching CUDA**. It bites when a task
inherits instead — a nested `@spawn` inside a job, or any helper task that
allocates — landing on an arbitrary device intermittently.

**MANDATORY design rules:**

1. Every worker task calls `CUDA.device!(assigned)` as its **first** CUDA action.
2. No job may spawn CUDA-touching subtasks without binding them explicitly.
3. **Assert** `CUDA.device() == assigned` at the top of every job. Cheap
   insurance against silent migration; failure mode is otherwise a
   nondeterministic wrong-device bug.

### Finding 3 — existing PEDANTIC_MATH already propagates (earlier worry retracted)

`GPUUtils.__init__` calls `math_mode!(PEDANTIC_MATH)`, commented as needed
because TF32 "destabilizes LBFGS line search and can cause early termination"
[source: utils/gpu_utils.jl:L14-L16].

Because that call sets `default_math_mode[]` at module load, and stickiness
propagates to later-created tasks [source: lib/cudadrv/state.jl:L32, L48-L51],
**workers inherit PEDANTIC_MATH for free**. Rev 1 of this spec claimed workers
would silently lose it — that was wrong.

Residual risk is only ordering: inheritance requires `__init__` to have run
before task creation, which module load guarantees in practice. Re-applying
per worker is harmless and makes the guarantee explicit rather than
load-order-dependent — do it as belt-and-braces, not as a bug fix.

## II.2 Device validation (`utils/gpu/device_pool.jl`)

### CUDA_VISIBLE_DEVICES renumbers devices

`CUDA_VISIBLE_DEVICES=2,5` makes CUDA see two devices indexed **0 and 1** — not
2 and 5. Two ID spaces: physical and visible.
[source: @architect — confirmed this session. Not read from CUDA.jl source the
way II.1 was; the evidence class here is operator attestation, which is
sufficient for a design decision but is not a code citation.]

**Design rule: the scheduler speaks only the visible space** — the only space
`CUDA.device!` accepts, sized by `CUDA.ndevices()`.

### Validation — eager, at construction, before any job runs

| # | Check | Failure example | Message must include |
|---|---|---|---|
| 1 | requested count <= `CUDA.ndevices()` | ask 8, have 4 | both numbers **and** current `CUDA_VISIBLE_DEVICES` |
| 2 | every explicit id in `0:ndevices()-1` | `[0,1,4]` with 4 visible | valid range; CUDA is 0-indexed, Julia 1-indexed |
| 3 | no duplicate ids | `[0,1,1]` | which id repeated (silently halves throughput) |
| 4 | GPU available at all | CPU-only box | **hard error** (Q3: no fallback) |
| 5 | per-device functional probe | device 2 in exclusive-compute mode | which device failed and why |

Check 1's env-var reporting is the highest-value part: the most confusing form
of this bug is 8 physical GPUs but only 4 visible because SLURM or a shell
export narrowed the set. Naming the env var turns a mystery into an obvious fix.

Check 5 is new: the current code probes `CUDA.functional()` once globally
[source: utils/gpu_utils.jl:L11]. On a multi-GPU box one device can be unusable
(ECC error, another process, exclusive-compute) while the rest are fine.

## II.3 Scheduler (`utils/gpu/scheduler.jl`)

Per Q2: **work-stealing queue, k=1 per device.** Push N jobs into a `Channel`,
spawn M workers, each binds via `CUDA.device!` once (per II.1 rules) then pulls
until drained. No barrier — a freed GPU takes the next job immediately.

This replaces, rather than tunes, the `Iterators.partition` batch barrier
[source: utils/two_d_grid_search_hyperparameters.jl:L176-L200], where every
batch waits on its slowest member. That matters because job costs genuinely
differ (`neuron_count=256` vs `=16` in one `run_sweep`
[source: utils/experiments.jl:L265-L272]).

**Generic over job type.** Roughly `run_staged(jobs, worker_fn; devices)`, job =
any closure. `run_transfer`, `run_sweep`, `run_all.jl`, `grid_search_2d` become
callers. Per the scope constraint, the scheduler never inspects a variant.

If k>1 is later enabled, `estimate_batch_size()` must do `CUDA.device!(i)`
before `CUDA.available_memory()` for per-device numbers
[source: utils/two_d_grid_search_hyperparameters.jl:L99-L111 — currently global].

## II.4 Determinism

Already correct per job: `initialize_network` derives from `settings.seed`
[source: architectures/PINN.jl:L177-L200]; `train_one` gives each model
`cfg.seed + seed_offset` [source: utils/experiments.jl:L153-L157]. Execution
order does not affect any individual model.

**One hazard.** `run_all.jl` calls global `Random.seed!(1234)`
[source: scripts/shared/run_all.jl:L21]. Under M concurrent workers the global
RNG is shared mutable state; jobs drawing from it get interleaving-dependent
values.

**Rule: a job owns its RNG.** `run_transfer` already does this right, passing
explicit `MersenneTwister(rng_seed + i)` per dataset
[source: utils/experiments.jl:L299-L305]. Make it mandatory.

**Stated limit.** Determinism is *per job*, not bitwise across heterogeneous
GPUs — FP reduction order can differ by device. Same job + same device =
reproducible. Record the device in job metadata so discrepancies are
diagnosable.

## II.5 Restart / resume

Machinery exists; only bookkeeping is missing.

- `train_pinn` accepts `snapshot_path`, warm-starts
  [source: architectures/PINN.jl:L401, L417-L419]
- `snapshot_epoch_interval` drives periodic saves
  [source: architectures/PINN.jl:L401]
- Snapshots are self-contained safetensors; `load_model` returns net, params,
  state, metadata with no external `PINNSettings`
  [source: utils/snapshot_utils.jl:L20-L27]
- Each job writes to its own tag-keyed dir with `model.safetensors`
  [source: utils/experiments.jl:L158-L167]

**Design: job manifest + idempotency.** Write the job list at scheduler start
(job id, variant, seed, config hash, output dir) to
`results/staged-{run_id}/manifest.json` (Q4). On restart: completed -> skip;
partial snapshot -> resume via `snapshot_path`; else -> run fresh. Tags are
deterministic functions of config [source: utils/experiments.jl:L153-L167], so
the same run yields the same tags, making skip-if-done safe.

Two details that decide whether this works:

1. **Completion marker distinct from "files exist."** A job killed mid-write
   leaves a truncated `model.safetensors` that looks done. Write `done.json`
   *after* the artifact closes; only that counts.
2. **Config hash in the manifest.** Editing `maxiters` then restarting would
   otherwise silently skip old-config jobs and produce a corrupt mixed result
   set. Compare on resume; refuse across a mismatch unless forced.

## II.6 Job-list builder (NEW)

The scheduler consumes a **flat, indexed** job list. Expanding axes into that
list is a separate concern — otherwise every experiment script regrows its own
nested loops.

`build_jobs(axes; fixed)` cross-products the staged axes, holds the rest fixed,
and assigns deterministic job ids. For this milestone the only staged axis is
variant (2 values); the builder must still be written generically, per the scope
constraint.

**Trainings, not evaluations, are the jobs.** `run_transfer` trains 6 models but
evaluates 36 cells [source: utils/experiments.jl:L296-L310]. Only trainings are
GPU jobs; evaluation is cheap and runs on the worker that trained the model.
Getting this wrong overschedules 6x.

### Axis inventory (for later milestones — NOT in scope now)

Already exist as sweeps: `neuron_count` and `N`
[source: utils/experiments.jl:L253-L262], region (6)
[source: scripts/shared/run_all.jl:L26-L27], shell radius
[source: utils/experiments.jl:L14-L16].

Exist as settings, never swept: `seed`, `num_points`, `n_per_region`,
`maxiters`, domain [source: utils/experiments.jl:L92-L95], `optimizer`
[source: architectures/PINN.jl:L81 — adam only; LBFGS disabled per
architectures/PINN.jl:L72].

@claude-opus-5 recommendation for the NEXT milestone: **seed replicates.**
Everything is currently n=1 per configuration, so "eigenvalue beats
power-series" cannot be separated from initialization luck. Seed plumbing
already exists [source: utils/experiments.jl:L153-L157], making this the
cheapest axis with the largest inferential payoff.

## II.7 Results collection

The strategy exists. Shapes return `PanelSet`, never a PNG — rendering is the
viewer's job [source: utils/experiments.jl:L26-L27], with
`save_experiment`/`load_experiment` as the JSON boundary
[source: utils/experiments.jl:L50-L61]. The index-into-preallocated-vector
pattern for out-of-order results is already in the grid search
[source: utils/two_d_grid_search_hyperparameters.jl:L160-L163].

**Scheduler contract:** a job returns a value; the scheduler places it at the
job's index; the caller assembles the PanelSet after the queue drains. The queue
stays generic over `Any` and never learns about PanelSpec. The sequential
`push!` accumulations in `run_sweep` [source: utils/experiments.jl:L269-L271]
become indexed writes.

**Plus, for resume:** partial results must be recoverable. If 40 of 50 jobs
finished before a crash, restart reloads those 40 from disk. So each job writes
its own small result file into its output dir, and the PanelSet is assembled
from those files — not held in memory until the full run completes.

---

# PART III — Variant Registry

## III.1 Representation is the ROOT, not a peer axis

@architect: variants start FROM the representation, because this is an
optimization problem on a defined loss, and the representation determines that
loss.

The code agrees. Representation determines four things that **must move
together**:

1. **Input encoding** — `canonicalize_alpha(vec(ode_matrix))` vs
   `tau_delta_from_alpha` -> `Float32[tau, delta]`
   [source: utils/snapshot_utils.jl:L28-L33]
2. **Output width/semantics** — N+1 monomial coefficients vs `(mu, k, A, B)`
   [source: architectures/PINN.jl:L85-L87, L115-L120]
3. **Loss triple** — `batched_power_series_losses` vs
   `batched_eigenvalue_losses` [source: utils/experiments.jl:L147-L150]
4. **Reconstruction of u(x)** — `evalpoly` over monomials vs
   `exp(mu*x)(A*C + B*S)` [source: utils/snapshot_utils.jl:L41-L51]

These cannot be mixed: a power-series output vector is meaningless to the
eigenvalue loss; eigenvalue reconstruction is undefined on N+1 monomial
coefficients. The existing comment already states this
[source: architectures/PINN.jl:L82-L87].

**Therefore the structure is a tree, not a product.** A free product would
permit combinations that do not typecheck, let alone converge.

```
representation (root — fixes encode / io_dims / loss form / reconstruct)
└── objective variant (which components optimized, how weighted)
    └── free parameters (neuron_count, seed, N, maxiters, trunk, optimizer)
```

## III.2 Loss FORM vs loss WEIGHTING

Representation fixes the loss **form** — which residuals exist, how computed.
It does NOT fix:

- **Weights.** `supervised_weight`/`pde_weight` are independent fields
  [source: architectures/PINN.jl:L78-L79]; the whole 2D grid search sweeps them
  [source: utils/two_d_grid_search_hyperparameters.jl:L112-L120]. Both
  representations have both.
- **Which components enter the objective.** `train_one` records
  `"objective_components" => "pde + supervised"` and
  `"diagnostic_components" => "bc"` [source: utils/experiments.jl:L163-L167] —
  BC is computed but excluded from the optimized objective. A choice, not a
  consequence of the representation.

Both are fixed for this milestone (see "Resolved Decisions").

## III.3 Why a registry: variant knowledge is currently non-local

One variant axis is expressed as **five separate conditionals in four files**:

| Decision | Site |
|---|---|
| io_dims | `Val` dispatch [source: architectures/PINN.jl:L115-L120] |
| buffer construction | ternary [source: architectures/PINN.jl:L429-L431] |
| buffer construction (again) | ternary [source: utils/experiments.jl:L126-L130] |
| loss selection | ternary [source: utils/experiments.jl:L147-L150] |
| reconstruction + encoding | branches [source: utils/snapshot_utils.jl:L28-L33, L41-L51] |

Adding a representation means finding all five. `io_dims`'s `Val` dispatch
[source: architectures/PINN.jl:L115] is already the registry pattern — applied
to exactly one of the five decisions.

Two variants also exist as **file copies** outside the system:
`PINN_RNN.jl` builds `Lux.Recurrence(Lux.GRUCell(1 => 64))`
[source: architectures/PINN_RNN.jl:L88-L89], and `PINN_specific.jl` defines a
second `module PINN` [source: architectures/PINN_specific.jl:L24]. Neither can
be selected, staged, or compared. Variants-by-file-copy is what a registry
replaces.

## III.4 Design

- A **representation** is a first-class entity supplying the four inseparable
  operations of III.1, plus `build_network`. Few of these (2 today). Adding one
  is mathematical work, not config.
- A **variant** is a named leaf: representation + objective spec + free params.
  Many of these; adding one should be a line in a table.
- Registry maps variant name -> `(representation, objective, params)`, with the
  representation validated as existing.

This preserves "variants start from the representation" **by construction**: a
variant cannot exist without naming a valid representation, and inherits the
four operations rather than choosing them.

**Do NOT add fields to `PINNSettings`.** It is already 15 positional fields with
a backwards-compat constructor shim for the last one added
[source: architectures/PINN.jl:L67-L98]; every new axis breaks positional call
sites again. Instead `PINNSettings` carries **one** symbol, `variant`, which
subsumes `representation` — fewer fields than today, not more.

**Mechanism: `Dict{Symbol,VariantSpec}`, not `Val`.** `Val` is type-stable but
compile-time-fixed; a Dict allows runtime registration and enumeration
(`keys(REGISTRY)` — needed by the job-list builder). Lookup happens once per
`train_pinn` call, not per iteration, so the dynamic dispatch cost is
irrelevant. Keep the existing `Val` dispatch where it already works
[source: architectures/PINN.jl:L115].

## III.5 Naming: `:eigenvalue` vs "trace-determinant"

@architect calls it the trace-determinant rep; the code calls it `:eigenvalue`
[source: architectures/PINN.jl:L87]. Both are half-right: its *input* encoding
is trace-determinant (`tau_delta_from_alpha`
[source: utils/snapshot_utils.jl:L29-L31], reinforced by `TRACE_DET_REGIONS` and
`alpha_from_tau_delta` [source: utils/plugboard.jl:L581]), while its *output* is
eigenvalue-flavored `(mu, k, A, B)` [source: architectures/PINN.jl:L86].

This is exactly how a codebase ends up with two names for one thing. The variant
name becomes a durable identifier — it goes into checkpoint metadata
[source: utils/safetensors_utils.jl:L100-L110] and into resume job keys — so it
is far cheaper to settle now than after result files exist under both names.

**RESOLVED (@architect): the variant name is `:eigenvalue`.**

Matches what is already in the field [source: architectures/PINN.jl:L87] and in
saved checkpoint metadata [source: utils/experiments.jl:L163-L167], so no
migration of existing result files is needed.

Requirement that follows: the variant's docstring MUST document that its *input*
encoding is trace-determinant (`tau_delta_from_alpha`
[source: utils/snapshot_utils.jl:L29-L31]) even though the variant is named for
its output. Without that note the half-right name becomes a trap for the next
reader, which is the exact failure this decision is meant to prevent.

## III.6 Knock-on benefit

`run_all.jl` hardcodes `for rep in [:power_series, :eigenvalue]`
[source: scripts/shared/run_all.jl:L45], so every experiment runs in exactly
both. Under a registry this becomes "iterate over these registered variants" —
more general, and exactly the job list the staged scheduler wants.

---

## Sequencing

1. **Part I — utils/ reorg.** Pure moves + include fixes + typo fix. Own commit.
2. **`utils/gpu/device_pool.jl`.** Visible-device enumeration, checks 1-5,
   per-device probe, explicit-binding rules from II.1.
3. **Thread `device` through** `GPUUtils.to_device` -> `train_pinn`, replacing
   the `Ref` singleton [source: utils/gpu_utils.jl:L9-L10]. Add the per-job
   device assertion (II.1 rule 3).
4. **Part III — variant registry.** Needed before the job-list builder can
   enumerate variants.
5. **`utils/gpu/scheduler.jl`** + **job-list builder.** Channel queue, manifest,
   resume, indexed results.
6. **Convert `run_transfer` first** — slowest shape at ~15 min
   [source: scripts/shared/run_all.jl:L11], cleanest job independence.
7. Convert remaining shapes, then `grid_search_2d`.
8. **Doc pass** over `docs/julia-modules/` (Q5).

## Open items — ALL RESOLVED (@architect)

| # | Item | Resolution |
|---|---|---|
| 1 | Weight values | **Both 1.0.** Already the default [source: utils/experiments.jl:L92-L95] — no config change. |
| 2 | Variant naming | **`:eigenvalue`.** Docstring must note the trace-determinant *input* (III.5). |
| 3 | CUDA_VISIBLE_DEVICES remapping | **Confirmed by @architect.** II.2 citation updated to operator attestation. |
| 4 | Throughput spike | **Deferred to post-implementation.** Does not block design. |

### Deferred to after implementation

The throughput spike: confirm M workers on M devices actually execute
concurrently rather than serializing, and that per-device memory reporting
works. This was originally raised as a *correctness* question; that part is now
answered from source in II.1, leaving only a performance question. If staged
throughput fails to scale with M, that is a tuning problem inside the existing
design, not a redesign.

**No open items block implementation. Part I may begin on @architect's go.**
