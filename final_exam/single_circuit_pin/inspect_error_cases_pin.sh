#!/usr/bin/env bash
# Read-only checkpoint/raw-data inspection. Edit these settings, then submit.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"

# Original experiment root, NOT the prediction_probe directory.
SOURCE_RUN="${SOURCE_RUN:-$HERE/results_pin/two_stage_opamp_otaf_lasd_moe_20261124T161718Z_1748571_30854_pin}"
CIRCUIT="two_stage_opamp_otaf"
SAMPLE_IDS=(41074 38711 38196)       # Original SQLite sample_id, NOT row number.
MODELS=(device_op_cascade_pin lasd_graph_layer_headwise_detach_pin_v3)
SEEDS=(29)
SPLIT="test"
RAW_DIR="${RAW_DIR:-$PROJECT_ROOT/runs/raw/full_data/$CIRCUIT}"
WAVEFORM_MANIFEST="${WAVEFORM_MANIFEST:-}" # Optional explicit sample-ID -> saved CSV mapping.
REFERENCE_SAMPLES=128               # Train-only OP error reference. 0 disables; no training.
REFERENCE_SEED=17
BATCH_SIZE=32
PLOTS=1
DEVICE="${DEVICE:-cuda}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-b_phzhwu}"
SLURM_PARTITION="${SLURM_PARTITION:-ex01A800}"
SLURM_CPUS="${SLURM_CPUS:-4}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_MEM="${SLURM_MEM:-32G}"
SLURM_TIME="${SLURM_TIME:-01:00:00}"
export JEPA_CONDA_ENV="${JEPA_CONDA_ENV-newbase}"
export JEPA_CONDA_HOME="${JEPA_CONDA_HOME:-/share/software/anaconda3}"
DRY_RUN=0
LOCAL=0
for argument in "$@"; do
    case "$argument" in
        --dry-run) DRY_RUN=1 ;;
        --local) LOCAL=1 ;;
        -h|--help)
            echo 'Usage: bash final_exam/single_circuit_pin/inspect_error_cases_pin.sh [--dry-run] [--local]'
            echo 'Edit SOURCE_RUN, CIRCUIT, SAMPLE_IDS, MODELS and SEEDS. Source IDs are never row positions.'
            echo 'No training, new sampling, normalization fit, simulator execution or source writes.'
            echo 'One job replays cases + bounded train OP references. Missing waveforms are explicitly reported.'
            exit 0 ;;
        *) echo "Unknown option: $argument" >&2; exit 2 ;;
    esac
done
(( ${#SAMPLE_IDS[@]} && ${#MODELS[@]} && ${#SEEDS[@]} )) || { echo 'IDs/models/seeds must be nonempty.' >&2; exit 2; }
[[ "$CIRCUIT" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'Invalid circuit name.' >&2; exit 2; }
for count in "$REFERENCE_SAMPLES" "$REFERENCE_SEED" "$SLURM_GPUS" "${SEEDS[@]}" "${SAMPLE_IDS[@]}"; do
    [[ "$count" =~ ^[0-9]+$ ]] || { echo 'Counts, IDs and seeds must be nonnegative integers.' >&2; exit 2; }
done
for count in "$BATCH_SIZE" "$SLURM_CPUS"; do
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo 'Batch size and CPUs must be positive.' >&2; exit 2; }
done
[[ "$PLOTS" == 0 || "$PLOTS" == 1 ]] || { echo 'PLOTS must be 0 or 1.' >&2; exit 2; }
if [[ "$DEVICE" == cuda* ]] && (( ! LOCAL && SLURM_GPUS < 1 )); then
    echo 'CUDA inspection requires a GPU.' >&2; exit 2
fi
cd "$PROJECT_ROOT"
export CIRCUIT_JEPA_ROOT="$PROJECT_ROOT" JEPA_PROJECT_ROOT="$PROJECT_ROOT" PYTHONUNBUFFERED=1
run_id="${CIRCUIT}_error_cases_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}_pin"
result_dir="$HERE/results_pin/$run_id"
log_dir="$HERE/logs_pin/$run_id"
hpc_dir="$HERE/hpc_logs_pin/$run_id"
arguments=(--source-run "$SOURCE_RUN" --circuit "$CIRCUIT" --sample-ids "${SAMPLE_IDS[@]}"
    --models "${MODELS[@]}" --seeds "${SEEDS[@]}" --split "$SPLIT" --raw-dir "$RAW_DIR"
    --reference-samples "$REFERENCE_SAMPLES" --reference-seed "$REFERENCE_SEED" --batch-size "$BATCH_SIZE"
    --device "$DEVICE" --output "$result_dir" --log-dir "$log_dir")
if [[ -n "$WAVEFORM_MANIFEST" ]]; then arguments+=(--waveform-manifest "$WAVEFORM_MANIFEST"); fi
if (( ! PLOTS )); then arguments+=(--no-plots); fi
runner=("$PROJECT_ROOT/scripts/run/slurm_run.sh" "${JEPA_PYTHON:-python}" -u
    "$PROJECT_ROOT/final_exam/error_case_probe_pin.py" "${arguments[@]}")
if (( LOCAL )); then
    command=(bash "${runner[@]}")
else
    options=(--parsable --job-name="error-cases-${CIRCUIT//_/-}-pin" --chdir="$PROJECT_ROOT"
        --output="$hpc_dir/%j_pin.out" --error="$hpc_dir/%j_pin.err"
        --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS"
        --mem="$SLURM_MEM" --time="$SLURM_TIME" --export=ALL)
    if (( SLURM_GPUS > 0 )); then options+=(--gres="gpu:$SLURM_GPUS"); fi
    if [[ -n "$SLURM_ACCOUNT" ]]; then options+=(--account="$SLURM_ACCOUNT"); fi
    if [[ -n "$SLURM_PARTITION" ]]; then options+=(--partition="$SLURM_PARTITION"); fi
    command=(sbatch "${options[@]}" "${runner[@]}")
fi
printf '%q ' "${command[@]}"
printf '\n'
if (( DRY_RUN )); then exit 0; fi
# Activate the same configured environment even for login-node stdlib preflight.
bash "$PROJECT_ROOT/scripts/run/slurm_run.sh" "${JEPA_PYTHON:-python}" -u \
    "$PROJECT_ROOT/final_exam/error_case_probe_pin.py" "${arguments[@]}" --inspect-source
if (( LOCAL )); then exec "${command[@]}"; fi
command -v sbatch >/dev/null || { echo 'sbatch unavailable; use --local in a compute allocation.' >&2; exit 1; }
mkdir -p "$HERE/hpc_logs_pin"
mkdir "$hpc_dir"
cp -- "$HERE/inspect_error_cases_pin.sh" "$hpc_dir/submitted_inspect_error_cases_pin.sh"
printf '%q ' "${command[@]}" > "$hpc_dir/submission_pin.txt"
printf '\n' >> "$hpc_dir/submission_pin.txt"
job_id="$("${command[@]}")"
printf 'Submitted error case probe job %s\nResults: %s\nExperiment logs: %s\nHPC logs: %s\n' \
    "$job_id" "$result_dir" "$log_dir" "$hpc_dir"
