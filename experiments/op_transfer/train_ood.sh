#!/usr/bin/env bash
# Edit circuit roles, models and seeds here, then run to submit one GPU job.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"

# ===== Experiment settings: edit these for your next run. =====
TRAIN_CIRCUITS=(5t_ota Opamp0 Opamp1 two_stage_opamp_otaf)
OOD_CIRCUITS=(two_stage_folded_opamp)
MODELS=(
  design_supervised
  op_oracle
  op_distill
  predicted_op
  dual_distill
  dual_design_control
  design_op_oracle
  student_predicted_op
  student_predicted_op_distill
)
# Focused cascade comparison: retain design_supervised, predicted_op,
# design_op_oracle and student_predicted_op above.
SEEDS=(17 29 43)
CONFIG="$HERE/example_config.json"
RAW_ROOT="$PROJECT_ROOT/runs/raw/full_data"
DEVICE="cuda"
# ============================================================

ACTION="--submit"
if (( $# > 1 )); then
    echo 'Usage: bash experiments/op_transfer/train_ood.sh [--dry-run | --help]' >&2
    exit 2
fi
case "${1:-}" in
    '') ;;
    --dry-run) ACTION="--dry-run" ;;
    -h|--help)
        echo 'Edit TRAIN_CIRCUITS, OOD_CIRCUITS, MODELS, SEEDS, CONFIG and RAW_ROOT above.'
        echo 'Run without arguments to submit; --dry-run previews without submitting.'
        exit 0 ;;
    *) echo "Unknown argument: $1; use --help" >&2; exit 2 ;;
esac
if (( ${#TRAIN_CIRCUITS[@]} == 0 || ${#OOD_CIRCUITS[@]} == 0 || ${#MODELS[@]} == 0 || ${#SEEDS[@]} == 0 )); then
    echo 'Set ID and OOD circuits, at least one model, and at least one seed.' >&2
    exit 2
fi
cd "$PROJECT_ROOT"
exec bash "$HERE/submit.sh" "$ACTION" \
  --train-circuits "${TRAIN_CIRCUITS[@]}" \
  --ood-circuits "${OOD_CIRCUITS[@]}" \
  --raw-root "$RAW_ROOT" \
  --config "$CONFIG" \
  --models "${MODELS[@]}" \
  --seeds "${SEEDS[@]}" \
  --device "$DEVICE"
