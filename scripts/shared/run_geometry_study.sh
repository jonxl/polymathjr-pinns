#!/usr/bin/env bash
# Train the bb2a102 geometry mirrors and refresh the local HTML study.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
study_date="${1:-$(date +%F)}"
study_root="${2:-results/geometry/dual-representation-${study_date}}"
iterations="${3:-3000}"
experiment_list="${4:-family_disks,spiral_transfer,pair_transfer,parabola_segment,parabola_slide,parabola_curves,parabola_bias,axis_transfer,axis_sweep}"
IFS=',' read -r -a experiments <<< "$experiment_list"
for experiment in "${experiments[@]}"; do
  case "$experiment" in
    family_disks|spiral_transfer|pair_transfer|parabola_segment|parabola_slide|parabola_curves|parabola_bias|axis_transfer|axis_sweep) ;;
    *) printf 'Unknown experiment: %s\n' "$experiment" >&2; exit 2 ;;
  esac
done

export_report() {
  if [[ -n "${STUDY_PYTHON:-}" ]]; then
    "$STUDY_PYTHON" scripts/charts/export_dual_study.py --date "$study_date" --geometry-root results/geometry/dual-representation-exp-2 --geometry-root "$study_root"
  else
    uv run scripts/charts/export_dual_study.py --date "$study_date" --geometry-root results/geometry/dual-representation-exp-2 --geometry-root "$study_root"
  fi
}

export_report
for experiment in "${experiments[@]}"; do
  "${JULIA_BIN:-julia}" --project=. scripts/shared/train_geometry_pair.jl "$study_root" "$experiment" "$iterations"
  export_report
done
