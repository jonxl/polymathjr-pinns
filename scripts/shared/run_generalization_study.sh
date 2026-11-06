#!/usr/bin/env bash
# All generalization suites, all three variants; refresh the command-center report.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=true; shift; fi
study_date="${1:-$(date +%F)}"
study_root="${2:-}"
iterations="${3:-1000000}"
[[ $# -le 3 && "$iterations" =~ ^[1-9][0-9]*$ ]] || { echo 'Usage: run_generalization_study.sh [--dry-run] [DATE] [NEW_OUTPUT_ROOT] [ITERATIONS]' >&2; exit 2; }
if [[ -z "$study_root" ]]; then
  day_root="results/generalization/${study_date}"
  run_prefix="run-$(date +%H%M%S)"
  if $dry_run; then
    study_root="${day_root}/${run_prefix}-PREVIEW"
  else
    mkdir -p -- "$day_root"
    # Atomically reserve an empty directory, even for simultaneous launches.
    study_root="$(mktemp -d "${day_root}/${run_prefix}-XXXXXX")"
  fi
fi
report_root="${study_root}/report"
train=("${JULIA_BIN:-julia}" --project=. -t auto scripts/generalization_variants.jl
  --experiments all --reps power_series,eigenvalue,power_series_variation
  --maxiters "$iterations" --output-root "$study_root")
if [[ -n "${STUDY_PYTHON:-}" ]]; then exporter=("$STUDY_PYTHON"); else exporter=(uv run); fi
report=("${exporter[@]}" scripts/charts/export_dual_study.py --date "$study_date"
  --output "$report_root" --geometry-root results/geometry/dual-representation-exp-2 --geometry-root "$study_root")
if $dry_run; then
  printf 'All 14 experiment types, including both generalization-radius modes: 15 groups × 3 variants = 45 jobs. No training started.\n'
  printf '%q ' "${train[@]}"; printf '\n'
  printf '%q ' "${report[@]}"; printf '\n'
  exit 0
fi
printf 'Training output: %s\nReport directory: %s\n' "$study_root" "$report_root"
"${train[@]}"
"${report[@]}"
printf 'Study complete: %s\n' "$study_root"
