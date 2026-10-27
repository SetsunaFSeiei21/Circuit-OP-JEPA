#!/usr/bin/env bash
# Edit the experiment settings below, then run this script to submit one job.
# Use --dry-run to preview without submitting or creating output directories.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Experiment settings: change these for the next run.
CIRCUIT="${CIRCUIT:-two_stage_opamp_otaf}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/default.json}"
MODELS=(
  zerosim_device
  zerosim_device_op
  zerosim_device_op_only
)
SEEDS=(17 29 43)

# This convenience launcher submits by default; the lower-level wrappers keep
# their preview-only default. A sourced cluster preset must not change this.
SUBMIT_DRY_RUN=0
if (( $# > 1 )); then
    echo 'Usage: bash scripts/run/train_single_circuit.sh [--dry-run | --help]' >&2
    exit 2
fi
case "${1:-}" in
    '') ;;
    --dry-run) SUBMIT_DRY_RUN=1 ;;
    -h|--help)
        echo 'Usage: bash scripts/run/train_single_circuit.sh [--dry-run | --help]'
        echo 'Edit CIRCUIT, CONFIG, MODELS and SEEDS above. Default: submit one Slurm job.'
        echo 'CIRCUIT and CONFIG may also be supplied as environment variables.'
        exit 0
        ;;
    *) echo "Unknown argument: $1; use --help" >&2; exit 2 ;;
esac

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/run/cluster.env.example"
export DRY_RUN="$SUBMIT_DRY_RUN"
export JEPA_RESULTS_ROOT="${JEPA_RESULTS_ROOT:-$PROJECT_ROOT/runs/gpu_full}"
export JEPA_LOG_DIR="${JEPA_LOG_DIR:-$PROJECT_ROOT/hpc_logs/gpu_full}"

exec bash "$PROJECT_ROOT/scripts/run/submit_raw_comparison.sh" \
  --input "$PROJECT_ROOT/runs/raw/full_data/$CIRCUIT" \
  --config "$CONFIG" \
  --models "${MODELS[@]}" \
  --max-table-rows none \
  --seeds "${SEEDS[@]}" \
  --device cuda
