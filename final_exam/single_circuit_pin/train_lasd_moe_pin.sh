#!/usr/bin/env bash
# Edit these settings, then run this script to submit one job per circuit.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"

# Experiment settings. PE layers only -> node-wise Top-2 MoE -> one final encoder/decoder.
CIRCUITS=(two_stage_folded_opamp)
MODELS=(
    device_op_cascade_pin # Original last-layer, single-decoder reference.
    lasd_moe_pin         # Hidden-state fusion and score paths both train the encoder.
    lasd_moe_detach_pin  # Detach only Score Dynamics inputs; hidden states remain live.
)
SEEDS=(17 29 43)
# lasd_moe_pin controls the node-wise layer-selection MoE, not a per-PE decoder.
# op_transfer.epochs.op_predictor/readout control the two training stages.
# Set dataset.max_samples_per_circuit (null/all or a positive count) and sampling_seed in CONFIG.
CONFIG="$PROJECT_ROOT/final_exam/configs/lasd_moe_pin.json"
RAW_ROOT="$PROJECT_ROOT/runs/raw/full_data"
DEVICE="cuda"
MAX_TABLE_ROWS="none"

# Slurm and Python environment settings.
SLURM_ACCOUNT="${SLURM_ACCOUNT:-b_phzhwu}"
SLURM_PARTITION="${SLURM_PARTITION:-ex01A800}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_MEM="${SLURM_MEM:-80G}"
SLURM_TIME="${SLURM_TIME:-10-00:00:00}"
SMOKE_TIME="00:30:00"
CONDA_ENV="${JEPA_CONDA_ENV:-newbase}"
CONDA_HOME="${JEPA_CONDA_HOME:-/share/software/anaconda3}"
SLURM_MAIL_USER="${SLURM_MAIL_USER:-}"
SLURM_MAIL_TYPE="${SLURM_MAIL_TYPE:-BEGIN,END,FAIL}"

DRY_RUN=0
SMOKE=0
model_preset=""
for argument in "$@"; do
    case "$argument" in
        --dry-run) DRY_RUN=1 ;;
        --smoke) SMOKE=1 ;;
        --headwise|--headwise-v1|--hyper-connection|--hyper-connection-grad|--global-stats|--graph-layer|--graph-layer-detach|--graph-layer-detach-variants)
            if [[ -n "$model_preset" && "$model_preset" != "$argument" ]]; then
                echo 'Select one model preset per submission.' >&2; exit 2
            fi
            model_preset="$argument" ;;
        -h|--help)
            echo 'Usage: bash final_exam/single_circuit_pin/train_lasd_moe_pin.sh [--headwise | --headwise-v1 | --hyper-connection | --hyper-connection-grad | --global-stats | --graph-layer | --graph-layer-detach | --graph-layer-detach-variants] [--dry-run] [--smoke]'
            echo 'Edit CIRCUITS, MODELS, SEEDS, CONFIG and cluster settings at the top.'
            echo '--headwise selects lasd_moe_headwise_pin and lasd_moe_headwise_detach_pin for the configured circuits and seeds.'
            echo '--headwise-v1 compares lasd_moe_headwise_pin_v1 and lasd_moe_headwise_detach_pin_v1, with a last-PE residual.'
            echo '--hyper-connection compares lasd_moe_headwise_detach_pin and lasd_moe_headwise_detach_hc_pin.'
            echo '--hyper-connection-grad compares lasd_moe_headwise_hc_pin and lasd_moe_headwise_detach_hc_pin; only score gradients differ.'
            echo '--global-stats compares plain headwise and global-context models, each with and without score detach (four models).'
            echo '--graph-layer compares the cascade, global-score MoE and lasd_graph_layer_headwise_pin (full gradients).'
            echo '--graph-layer-detach runs only lasd_graph_layer_headwise_detach_pin; PE fusion values keep gradients.'
            echo '--graph-layer-detach-variants runs exactly graph detach v1/v2/v3: detached PE fusion taps; v2/v3 omit the final encoder; v3 adds the live last PE residual.'
            echo 'Graph/layer routing uses circuit-masked graph blocks, then per-node depth attention and an H-logit MLP.'
            echo 'Global context uses per-graph/per-PE/per-head mean R, std R, std T and their differences.'
            echo 'Each circuit submits an independent job. --dry-run creates no output directories.'
            echo '--smoke shortens training; dataset.max_samples_per_circuit controls sample count.'
            exit 0 ;;
        *) echo "Unknown option: $argument; use --help" >&2; exit 2 ;;
    esac
done
case "$model_preset" in
    --headwise) MODELS=(lasd_moe_headwise_pin lasd_moe_headwise_detach_pin) ;;
    --headwise-v1) MODELS=(lasd_moe_headwise_pin_v1 lasd_moe_headwise_detach_pin_v1) ;;
    --hyper-connection) MODELS=(lasd_moe_headwise_detach_pin lasd_moe_headwise_detach_hc_pin) ;;
    --hyper-connection-grad) MODELS=(lasd_moe_headwise_hc_pin lasd_moe_headwise_detach_hc_pin) ;;
    --global-stats) MODELS=(lasd_moe_headwise_pin lasd_moe_headwise_global_pin
                           lasd_moe_headwise_detach_pin lasd_moe_headwise_detach_global_pin) ;;
    --graph-layer) MODELS=(device_op_cascade_pin lasd_moe_headwise_global_pin lasd_graph_layer_headwise_pin) ;;
    --graph-layer-detach) MODELS=(lasd_graph_layer_headwise_detach_pin) ;;
    --graph-layer-detach-variants) MODELS=(lasd_graph_layer_headwise_detach_pin_v1
                                         lasd_graph_layer_headwise_detach_pin_v2
                                         lasd_graph_layer_headwise_detach_pin_v3) ;;
esac
(( ${#CIRCUITS[@]} && ${#MODELS[@]} && ${#SEEDS[@]} )) || {
    echo 'CIRCUITS, MODELS and SEEDS must be nonempty.' >&2; exit 2;
}
[[ "$SLURM_CPUS" =~ ^[1-9][0-9]*$ && "$SLURM_GPUS" =~ ^[0-9]+$ ]] || {
    echo 'Invalid CPU/GPU request.' >&2; exit 2;
}
if [[ "$DEVICE" == cuda* ]] && (( SLURM_GPUS < 1 )); then
    echo 'CUDA training requires at least one requested GPU.' >&2
    exit 2
fi
for seed in "${SEEDS[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
done
if (( SMOKE )); then SLURM_TIME="$SMOKE_TIME"; fi
if (( ! DRY_RUN )); then
    [[ -f "$CONFIG" ]] || { echo "Configuration file missing: $CONFIG" >&2; exit 1; }
    command -v sbatch >/dev/null || { echo 'sbatch is unavailable.' >&2; exit 1; }
fi
# Validate every input before submitting the first job.
declare -A checked_circuits=()
for circuit in "${CIRCUITS[@]}"; do
    [[ "$circuit" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
        echo "Circuit must be a directory name: $circuit" >&2; exit 2;
    }
    [[ -z "${checked_circuits[$circuit]:-}" ]] || {
        echo "Duplicate circuit: $circuit" >&2; exit 2;
    }
    checked_circuits[$circuit]=1
    if (( ! DRY_RUN )); then
        [[ -f "$RAW_ROOT/$circuit/sampling_history.sqlite3" ]] || {
            echo "Raw source missing: $RAW_ROOT/$circuit/sampling_history.sqlite3" >&2; exit 1;
        }
    fi
done

cd "$PROJECT_ROOT"
export JEPA_PROJECT_ROOT="$PROJECT_ROOT"
export JEPA_CONDA_ENV="$CONDA_ENV" JEPA_CONDA_HOME="$CONDA_HOME"
export PYTHONUNBUFFERED=1
for circuit in "${CIRCUITS[@]}"; do
    run_id="${circuit}_lasd_moe_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}_pin"
    hpc_dir="$HERE/hpc_logs_pin/$run_id"
    log_dir="$HERE/logs_pin/$run_id"
    result_dir="$HERE/results_pin/$run_id"
    options=(
        --parsable --job-name="final-lasd-moe-${circuit//_/-}-pin" --chdir="$PROJECT_ROOT"
        --output="$hpc_dir/%j_pin.out" --error="$hpc_dir/%j_pin.err"
        --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS"
        --mem="$SLURM_MEM" --time="$SLURM_TIME" --export=ALL
    )
    if (( SLURM_GPUS > 0 )); then options+=(--gres="gpu:$SLURM_GPUS"); fi
    if [[ -n "$SLURM_ACCOUNT" ]]; then options+=(--account="$SLURM_ACCOUNT"); fi
    if [[ -n "$SLURM_PARTITION" ]]; then options+=(--partition="$SLURM_PARTITION"); fi
    if [[ -n "$SLURM_MAIL_USER" ]]; then
        options+=(--mail-user="$SLURM_MAIL_USER" --mail-type="$SLURM_MAIL_TYPE")
    fi
    command=(sbatch "${options[@]}" "$HERE/run_pin.sh"
        --circuit "$circuit" --run-id "$run_id" --raw-root "$RAW_ROOT" --config "$hpc_dir/submitted_config_pin.json"
        --models "${MODELS[@]}" --seeds "${SEEDS[@]}" --device "$DEVICE"
        --max-table-rows "$MAX_TABLE_ROWS")
    if (( SMOKE )); then command+=(--smoke); fi
    printf '%q ' "${command[@]}"
    printf '\n'
    if (( DRY_RUN )); then continue; fi
    mkdir -p "$HERE/hpc_logs_pin"
    mkdir "$hpc_dir"
    cp -- "$CONFIG" "$hpc_dir/submitted_config_pin.json"
    if job_id="$("${command[@]}")"; then
        printf 'Submitted job %s\nHPC logs: %s\nExperiment logs: %s\nResults: %s\n' \
            "$job_id" "$hpc_dir" "$log_dir" "$result_dir"
        printf 'job_id=%s\nrun_id=%s\nconfig_source=%s\n' "$job_id" "$run_id" "$CONFIG" > "$hpc_dir/submission_pin.txt"
    else
        status=$?
        printf 'Submission failed with exit code %s. HPC directory: %s\n' "$status" "$hpc_dir" >&2
        exit "$status"
    fi
done
