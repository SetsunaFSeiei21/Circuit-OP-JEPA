#!/usr/bin/env bash
# Edit these settings. Runs locally on saved reports; no training/GPU/data load.
set -euo pipefail
SOURCE_RUN="${SOURCE_RUN:-}"  # Run root (containing training_pin) or training_pin itself.
K="${K:-2}"
MODELS_BETTER=(
    width_decoder_pin
    width_decoder_op_pin
)
MODELS_WORSE=(
    device_op_cascade_pin
)
OUTPUT="${OUTPUT:-}"  # Empty: <training_pin>/summary_seed_selection_k<K>_pin.xlsx
RECOVER_EXPORT=0     # Set 1 only when the source run status is failed_export.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
case "${1:-}" in
    -h|--help)
        echo 'Edit SOURCE_RUN, K, MODELS_BETTER and MODELS_WORSE at the top, then run without arguments.'
        echo 'Ranking: test macro R2. Unlisted models keep all seeds; outputs explicitly label selection.'
        echo 'SOURCE_RUN, K and OUTPUT also accept environment overrides. Existing files are never overwritten.'
        exit 0 ;;
esac
(( $# == 0 )) || { echo 'Edit the script settings; use --help.' >&2; exit 2; }
[[ -n "$SOURCE_RUN" ]] || { echo 'Set SOURCE_RUN to your saved result directory.' >&2; exit 2; }
[[ "$K" =~ ^[1-9][0-9]*$ ]] || { echo 'K must be a positive integer.' >&2; exit 2; }
[[ "$RECOVER_EXPORT" == 0 || "$RECOVER_EXPORT" == 1 ]] || { echo 'RECOVER_EXPORT must be 0 or 1.' >&2; exit 2; }
command=("${JEPA_PYTHON:-python}" "$PROJECT_ROOT/final_exam/summarize_selected_seeds_pin.py"
    "$SOURCE_RUN" --k "$K" --models-better "${MODELS_BETTER[@]}"
    --models-worse "${MODELS_WORSE[@]}")
if [[ -n "$OUTPUT" ]]; then command+=(--output "$OUTPUT"); fi
if (( RECOVER_EXPORT )); then command+=(--recover-export); fi
exec "${command[@]}"
