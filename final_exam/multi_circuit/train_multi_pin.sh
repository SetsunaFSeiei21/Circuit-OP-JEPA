#!/usr/bin/env bash
# Edit settings here. A model is trained jointly on ALL selected circuits.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
CIRCUITS=(two_stage_folded_opamp Opamp1)
MODELS=(
    mlp_pin
    xgboost_pin
    gat_pin
    deepgen_supervised_pin
    deepgen_op_pretrained_pin
    gtn_pin
    zerosim_supervised_pin
    accformer_pin
    device_op_cascade_pin
    # width_decoder_pin
    # lasd_graph_layer_headwise_detach_pin
    # lasd_graph_layer_headwise_detach_pin_v1
    # lasd_graph_layer_headwise_detach_pin_v2
    # lasd_graph_layer_headwise_detach_pin_v3
)
SEARCH_SEED=11
SEEDS=(17 29 43 59 71)
MODE="all"                           # all / search / train
FIXED_CONFIG=0                       # train only: bypass HPO with model JSONs
CAMPAIGN=""                          # Resume path: results_pin/multi_..._pin; saved settings win
CONFIG_DIR="$HERE/configs_pin"       # models/<name>.json + search_spaces/<name>.json
PROTOCOL="$CONFIG_DIR/protocol_pin.json"  # sampling count/seed, split, normalization
RAW_ROOT="$PROJECT_ROOT/runs/raw/full_data"
DEVICE="cuda"
MAX_CONCURRENT=2                     # GPU jobs, each executes one model's trials/seeds serially
SLURM_ACCOUNT="${SLURM_ACCOUNT:-b_phzhwu}"
SLURM_PARTITION="${SLURM_PARTITION:-ex01A800}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_MEM="${SLURM_MEM:-80G}"
SLURM_TIME="${SLURM_TIME:-10-00:00:00}"
export JEPA_CONDA_ENV="${JEPA_CONDA_ENV-newbase}"
export JEPA_CONDA_HOME="${JEPA_CONDA_HOME:-/share/software/anaconda3}"
# Submission planning uses stdlib only. Compute jobs activate the environment above.
PYTHON="${JEPA_SUBMIT_PYTHON:-python3}"
args=(--circuits "${CIRCUITS[@]}" --models "${MODELS[@]}" --seeds "${SEEDS[@]}"
    --search-seed "$SEARCH_SEED" --mode "$MODE" --config-dir "$CONFIG_DIR" --protocol "$PROTOCOL"
    --raw-root "$RAW_ROOT" --device "$DEVICE" --max-concurrent "$MAX_CONCURRENT"
    --account "$SLURM_ACCOUNT" --partition "$SLURM_PARTITION" --cpus "$SLURM_CPUS"
    --mem "$SLURM_MEM" --time "$SLURM_TIME")
if [[ -n "$CAMPAIGN" ]]; then args+=(--campaign "$CAMPAIGN"); fi
if (( FIXED_CONFIG )); then args+=(--fixed-config); fi
cd "$PROJECT_ROOT"
exec "$PYTHON" "$HERE/launch_pin.py" "${args[@]}" "$@"
