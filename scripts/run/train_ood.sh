#!/usr/bin/env bash
# Submit one joint-training/OOD job; import and training run on the compute node.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Edit circuit roles, model selection and seeds here. Names must match SQLite.
TRAIN_CIRCUITS=(5t_ota Opamp0 Opamp1)
OOD_CIRCUITS=("${OOD_CIRCUIT:-two_stage_folded_opamp}")
RAW_ROOT="${RAW_ROOT:-$PROJECT_ROOT/runs/raw/full_data}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/default.json}"
MODELS=(
  zerosim_pin
  zerosim_device
  jepa_zerosim
  zerosim_pin_op
  zerosim_device_op
  zerosim_pin_op_only
  zerosim_device_op_only
)
SEEDS=(17 29 43)

# Like train_single_circuit.sh, this launcher submits unless --dry-run is given.
SUBMIT_DRY_RUN=0
if (( $# > 1 )); then
    echo 'Usage: bash scripts/run/train_ood.sh [--dry-run | --help]' >&2
    exit 2
fi
case "${1:-}" in
    '') ;;
    --dry-run) SUBMIT_DRY_RUN=1 ;;
    -h|--help)
        echo 'Usage: bash scripts/run/train_ood.sh [--dry-run | --help]'
        echo 'Edit TRAIN_CIRCUITS, OOD_CIRCUITS, CONFIG, MODELS and SEEDS above.'
        echo 'Default: submit one job; train on 5t_ota/Opamp0/Opamp1, hold out two_stage_folded_opamp.'
        echo 'OOD_CIRCUIT, RAW_ROOT and CONFIG may also be supplied as environment variables.'
        exit 0
        ;;
    *) echo "Unknown argument: $1; use --help" >&2; exit 2 ;;
esac

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/run/cluster.env.example"
export DRY_RUN="$SUBMIT_DRY_RUN"
export JEPA_RESULTS_ROOT="${JEPA_RESULTS_ROOT:-$PROJECT_ROOT/runs/gpu_full}"
export JEPA_LOG_DIR="${JEPA_LOG_DIR:-$PROJECT_ROOT/hpc_logs/gpu_full}"
export JEPA_JOB_KIND="${JEPA_JOB_KIND:-ood_run}"

INPUTS=()
for circuit in "${TRAIN_CIRCUITS[@]}" "${OOD_CIRCUITS[@]}"; do
    INPUTS+=("$RAW_ROOT/$circuit")
done

exec bash "$PROJECT_ROOT/scripts/run/submit_raw_comparison.sh" \
  --input "${INPUTS[@]}" \
  --train-circuits "${TRAIN_CIRCUITS[@]}" \
  --ood-circuits "${OOD_CIRCUITS[@]}" \
  --config "$CONFIG" \
  --models "${MODELS[@]}" \
  --max-table-rows none \
  --seeds "${SEEDS[@]}" \
  --device cuda
