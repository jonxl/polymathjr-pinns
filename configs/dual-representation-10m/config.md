# Dual-representation 10M study

The training run is defined by `scripts/shared/run_all.jl`. At startup it writes
the authoritative `config.md`, `config.json`, and split manifests into
`results/dual-representation-10m/`.

- 10,000,000 deterministic canonical training ODEs
- 500,000 validation ODEs used for checkpoint selection
- 1,000,000 final test ODEs, untouched until model selection is complete
- 8,192 ODEs per minibatch
- 100 epochs
- Checkpoints every 5 completed epochs
- Power-series degree `N=20` and eigenvalue representation trained from the
  identical canonical ODE indices and identical minibatch order
- Both dense MLPs consume the identical `(tau, delta)` input; the fixed monic
  coefficient of `u''` is reconstructed internally as `[delta, -tau, 1]`
- Power-series MLP: `2→64→64→64→21` (9,877 parameters)
- Eigenvalue MLP: `2→64→64→64→4` (8,772 parameters)
- Dataset generation, epoch shuffling, and model initialization use three
  independently configurable and recorded seeds
- Two global models and twelve family-specific models
- Eight GPUs used concurrently, with one worker and at most one active model
  per GPU; remaining model jobs wait in a shared queue

Each checkpoint records its dataset identifier, representation, scope, family,
epoch, optimizer-update count, example exposures, batch size, and series degree.
Final checkpoint filenames expose model type and parameter count:
`model-dense-mlp-p009877.checkpoint` for power series and
`model-dense-mlp-p008772.checkpoint` for eigenvalue models.
