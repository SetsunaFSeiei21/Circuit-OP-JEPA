#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${JEPA_PYTHON:-python}"
OUT="runs/comparison_smoke/$(date -u +%Y%m%dT%H%M%SZ)-$$"
if (( $# > 0 )) && [[ "$1" != --* ]]; then OUT="$1"; shift; fi
RUN_TESTS=0
OPTIONS=()
for argument in "$@"; do
    case "$argument" in
        --list-models) exec "$PYTHON" -m device_jepa --list-models ;;
        --run-tests) RUN_TESTS=1 ;;
        *) OPTIONS+=("$argument") ;;
    esac
done
[[ ! -e "$OUT" ]] || { echo 'Choose a new smoke output directory' >&2; exit 2; }
if [[ "$RUN_TESTS" == 1 ]]; then "$PYTHON" -m unittest discover -s tests -v; fi
mkdir -p "$OUT"
"$PYTHON" -m device_jepa make_synthetic --output "$OUT/data.json" --per-circuit 24
bash scripts/run/run_comparison.sh --data "$OUT/data.json" --config configs/smoke.json --output "$OUT/experiment" --seeds 17 --smoke "${OPTIONS[@]}"
