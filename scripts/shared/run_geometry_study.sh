#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
study_date="${1:-$(date +%F)}"
study_root="${2:-}"
iterations="${3:-}"
experiments="${4:-family_disks,spiral_transfer,pair_transfer,parabola_segment,parabola_slide,parabola_curves,parabola_bias,axis_transfer,axis_sweep}"
args=(--date "$study_date" --experiments "$experiments" --report)
if [[ -n "$iterations" ]]; then
  args+=(--maxiters "$iterations")
elif [[ -z "${STUDY_CONFIG:-}" ]]; then
  args+=(--maxiters 3000)
fi
[[ -z "$study_root" ]] || args+=(--output-root "$study_root")
[[ -z "${STUDY_CONFIG:-}" ]] || args+=(--config "$STUDY_CONFIG")
exec "${JULIA_BIN:-julia}" --project=. -t auto scripts/generalization_variants.jl "${args[@]}"
