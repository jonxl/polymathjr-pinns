#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
dry_run=()
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=(--dry-run); shift; fi
study_date="${1:-$(date +%F)}"
study_root="${2:-}"
iterations="${3:-${STUDY_ITERATIONS:-}}"
[[ $# -le 3 && ( -z "$iterations" || "$iterations" =~ ^[1-9][0-9]*$ ) ]] || { echo 'Usage: run_generalization_study.sh [--dry-run] [DATE] [NEW_OUTPUT_ROOT] [ITERATIONS]' >&2; exit 2; }
args=(--date "$study_date" --report)
if [[ -n "$iterations" ]]; then
  args+=(--maxiters "$iterations")
elif [[ -z "${STUDY_CONFIG:-}" ]]; then
  args+=(--maxiters 10000)
fi
for setting in 'STUDY_LANES:--lanes:^[1-9][0-9]*$' 'STUDY_DEGREES:--degrees:^[1-9][0-9]*(,[1-9][0-9]*)*$'; do
  IFS=: read -r name flag pattern <<< "$setting"
  value="${!name:-}"
  [[ -n "$value" ]] || continue
  [[ "$value" =~ $pattern ]] || { echo "$name has an invalid value: $value" >&2; exit 2; }
  args+=("$flag" "$value")
done
[[ -z "$study_root" ]] || args+=(--output-root "$study_root")
[[ -z "${STUDY_CONFIG:-}" ]] || args+=(--config "$STUDY_CONFIG")
exec "${JULIA_BIN:-julia}" --project=. -t auto scripts/generalization_variants.jl "${args[@]}" "${dry_run[@]}"
