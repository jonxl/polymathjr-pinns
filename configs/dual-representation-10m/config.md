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
- Two global models and twelve family-specific models

Each checkpoint records its dataset identifier, representation, scope, family,
epoch, optimizer-update count, example exposures, batch size, and series degree.
