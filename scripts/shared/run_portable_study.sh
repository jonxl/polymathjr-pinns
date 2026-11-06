#!/usr/bin/env bash
set -euo pipefail

# One-command lifecycle for the portable dual-representation study. Every
# stage is restart-safe: compatible completed models, checkpoint selection,
# and final-test results are reused; incompatible artifacts fail closed.

: "${DUAL_DATASET_SEED:=10001}"
: "${DUAL_SHUFFLE_SEED:=40001}"
: "${DUAL_MODEL_SEED:=1234}"
: "${DUAL_BATCH_SIZE:=16384}"
: "${DUAL_GPU_COUNT:=8}"
: "${JULIA_BIN:=julia}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
# An explicit root retains the existing validated restart behavior.
if [[ -z "${DUAL_STUDY_ROOT:-}" ]]; then
  DUAL_STUDY_ROOT="$("${JULIA_BIN}" --project=. scripts/shared/new_run_directory.jl)"
fi

export DUAL_DATASET_SEED DUAL_SHUFFLE_SEED DUAL_MODEL_SEED DUAL_BATCH_SIZE
export DUAL_GPU_COUNT DUAL_STUDY_ROOT

run_name="dataset-${DUAL_DATASET_SEED}-shuffle-${DUAL_SHUFFLE_SEED}-model-${DUAL_MODEL_SEED}-batch-${DUAL_BATCH_SIZE}"
run_root="${DUAL_STUDY_ROOT}/${run_name}"

"${JULIA_BIN}" --project=. -t "${DUAL_GPU_COUNT}" scripts/shared/run_all.jl
"${JULIA_BIN}" --project=. scripts/shared/evaluate_study.jl select "${run_root}" "${DUAL_BATCH_SIZE}"
"${JULIA_BIN}" --project=. scripts/shared/evaluate_study.jl infer "${run_root}" "${DUAL_BATCH_SIZE}"
"${JULIA_BIN}" --project=. scripts/shared/evaluate_study.jl suite "${run_root}"
"${JULIA_BIN}" --project=. scripts/charts/dual_representation_study.jl "${run_root}"
"${JULIA_BIN}" --project=. scripts/shared/bundle_study.jl finalize "${run_root}"

printf 'Portable study complete: %s\n' "${run_root}"
