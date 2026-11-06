#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
dry_run=()
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=(--dry-run); shift; fi
study_date="${1:-$(date +%F)}"
study_root="${2:-}"
iterations="${3:-}"
[[ $# -le 3 && ( -z "$iterations" || "$iterations" =~ ^[1-9][0-9]*$ ) ]] || { echo 'Usage: run_generalization_study.sh [--dry-run] [DATE] [NEW_OUTPUT_ROOT] [ITERATIONS]' >&2; exit 2; }
args=(--date "$study_date" --report)
if [[ -n "$iterations" ]]; then
  args+=(--maxiters "$iterations")
elif [[ -z "${STUDY_CONFIG:-}" ]]; then
  args+=(--maxiters 1000000)
fi
[[ -z "$study_root" ]] || args+=(--output-root "$study_root")
[[ -z "${STUDY_CONFIG:-}" ]] || args+=(--config "$STUDY_CONFIG")
exec "${JULIA_BIN:-julia}" --project=. -t auto scripts/generalization_variants.jl "${args[@]}" "${dry_run[@]}"
