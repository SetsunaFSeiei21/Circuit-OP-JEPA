#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${JEPA_PYTHON:-python}"
OUT="${1:-runs/comparison_smoke/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
if (( $# > 1 )); then echo 'Usage: run_smoke.sh [OUTPUT_ROOT]' >&2; exit 2; fi
[[ ! -e "$OUT" ]] || { echo 'Choose a new smoke output directory' >&2; exit 2; }
"$PYTHON" -m unittest discover -s tests -v
mkdir -p "$OUT"
"$PYTHON" -m device_jepa make_synthetic --output "$OUT/data.json" --per-circuit 24
bash scripts/run/run_comparison.sh --data "$OUT/data.json" --config configs/smoke.json --output "$OUT/experiment" --seeds 17 --smoke
