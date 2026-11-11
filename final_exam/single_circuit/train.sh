#!/usr/bin/env bash
# Edit these settings, then run this script to submit one job per circuit.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"

# Experiment settings.
CIRCUITS=(Opamp1)
MODELS=(
    mlp xgboost gat deepgen_supervised deepgen_op_pretrained gtn
    zerosim_supervised accformer
    predicted_op student_predicted_op student_predicted_op_distill
    op_oracle design_op_oracle
)
SEEDS=(17 29 43)
# Set dataset.max_samples_per_circuit (null/all or a positive count) and sampling_seed in CONFIG.
CONFIG="$PROJECT_ROOT/final_exam/configs/default.json"
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
for argument in "$@"; do
    case "$argument" in
        --dry-run) DRY_RUN=1 ;;
        --smoke) SMOKE=1 ;;
        -h|--help)
            echo 'Usage: bash final_exam/single_circuit/train.sh [--dry-run] [--smoke]'
            echo 'Edit CIRCUITS, MODELS, SEEDS, CONFIG and cluster settings at the top.'
            echo 'Each circuit submits an independent job. --dry-run creates no output directories.'
            echo '--smoke shortens training; dataset.max_samples_per_circuit controls sample count.'
            exit 0 ;;
        *) echo "Unknown option: $argument; use --help" >&2; exit 2 ;;
    esac
done
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
    run_id="${circuit}_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}"
    hpc_dir="$HERE/hpc_logs/$run_id"
    log_dir="$HERE/logs/$run_id"
    result_dir="$HERE/results/$run_id"
    options=(
        --parsable --job-name="final-single-${circuit//_/-}" --chdir="$PROJECT_ROOT"
        --output="$hpc_dir/%j.out" --error="$hpc_dir/%j.err"
        --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS"
        --mem="$SLURM_MEM" --time="$SLURM_TIME" --export=ALL
    )
    if (( SLURM_GPUS > 0 )); then options+=(--gres="gpu:$SLURM_GPUS"); fi
    if [[ -n "$SLURM_ACCOUNT" ]]; then options+=(--account="$SLURM_ACCOUNT"); fi
    if [[ -n "$SLURM_PARTITION" ]]; then options+=(--partition="$SLURM_PARTITION"); fi
    if [[ -n "$SLURM_MAIL_USER" ]]; then
        options+=(--mail-user="$SLURM_MAIL_USER" --mail-type="$SLURM_MAIL_TYPE")
    fi
    command=(sbatch "${options[@]}" "$HERE/run.sh"
        --circuit "$circuit" --run-id "$run_id" --raw-root "$RAW_ROOT" --config "$hpc_dir/submitted_config.json"
        --models "${MODELS[@]}" --seeds "${SEEDS[@]}" --device "$DEVICE"
        --max-table-rows "$MAX_TABLE_ROWS")
    if (( SMOKE )); then command+=(--smoke); fi
    printf '%q ' "${command[@]}"
    printf '\n'
    if (( DRY_RUN )); then continue; fi
    mkdir -p "$HERE/hpc_logs"
    mkdir "$hpc_dir"
    cp -- "$CONFIG" "$hpc_dir/submitted_config.json"
    if job_id="$("${command[@]}")"; then
        printf 'Submitted job %s\nHPC logs: %s\nExperiment logs: %s\nResults: %s\n' \
            "$job_id" "$hpc_dir" "$log_dir" "$result_dir"
        printf 'job_id=%s\nrun_id=%s\nconfig_source=%s\n' "$job_id" "$run_id" "$CONFIG" > "$hpc_dir/submission.txt"
    else
        status=$?
        printf 'Submission failed with exit code %s. HPC directory: %s\n' "$status" "$hpc_dir" >&2
        exit "$status"
    fi
done
