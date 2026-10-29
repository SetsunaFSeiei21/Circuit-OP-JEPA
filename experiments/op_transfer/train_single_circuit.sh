#!/usr/bin/env bash
# Edit the settings below, then run this file to submit one full-data GPU job.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"

# ===== Experiment settings: edit these for your next run. =====
CIRCUIT="Opamp1"
# MODELS=(
#   design_supervised
#   op_oracle
#   op_distill
#   predicted_op
#   dual_distill
#   dual_design_control
#   design_op_oracle
#   student_predicted_op
# )
MODELS=(
  design_supervised
  op_oracle
  design_op_oracle
  predicted_op
  student_predicted_op
  student_predicted_op_distill
)
# Focused cascade comparison: retain design_supervised, predicted_op,
# design_op_oracle and student_predicted_op above.
# SEEDS=(17 29 43)
SEEDS=(17)
CONFIG="$HERE/example_config.json"
RAW_ROOT="$PROJECT_ROOT/runs/raw/full_data"
DEVICE="cuda"
# ============================================================

ACTION="--submit"
if (( $# > 1 )); then
    echo 'Usage: bash experiments/op_transfer/train_single_circuit.sh [--dry-run | --help]' >&2
    exit 2
fi
case "${1:-}" in
    '') ;;
    --dry-run) ACTION="--dry-run" ;;
    -h|--help)
        echo 'Edit CIRCUIT, MODELS, SEEDS, CONFIG and RAW_ROOT at the top of this file.'
        echo 'Run without arguments to submit; --dry-run previews without submitting.'
        exit 0 ;;
    *) echo "Unknown argument: $1; use --help" >&2; exit 2 ;;
esac
if [[ -z "$CIRCUIT" ]] || (( ${#MODELS[@]} == 0 || ${#SEEDS[@]} == 0 )); then
    echo 'Set a circuit, at least one model, and at least one seed.' >&2
    exit 2
fi
cd "$PROJECT_ROOT"
exec bash "$HERE/submit.sh" "$ACTION" \
  --circuit "$CIRCUIT" \
  --raw-root "$RAW_ROOT" \
  --config "$CONFIG" \
  --models "${MODELS[@]}" \
  --seeds "${SEEDS[@]}" \
  --device "$DEVICE"
