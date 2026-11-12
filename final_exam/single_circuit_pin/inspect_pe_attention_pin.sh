#!/usr/bin/env bash
# Edit this block, then run: bash final_exam/single_circuit_pin/inspect_pe_attention_pin.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"

# Inspection settings. SOURCE_RUN accepts the result root or its training_pin folder.
SOURCE_RUN="${SOURCE_RUN:-}"             # Fill in your saved result path here.
CIRCUIT="two_stage_folded_opamp"
N_SAMPLES=20                            # Cap at the available saved split size.
# Native ungated checkpoint; also supports zerosim_supervised_pin and ungated OP cascades.
MODELS=(device_op_cascade_pin)
SEED=17                                 # Training/checkpoint seed, not a new training run.
SPLIT="test"                            # train / val / test from the saved manifest.
SAMPLE_SEED=17                          # Fixed sample selection, shared by all MODELS.
BATCH_SIZE=8
VIEWER_SAMPLES=20                       # HTML cap only; CSV includes every sampled row.
DEVICE="${DEVICE:-cuda}"

# Slurm / Python. The default submits inference to a compute node, not the login node.
SLURM_ACCOUNT="${SLURM_ACCOUNT:-b_phzhwu}"
SLURM_PARTITION="${SLURM_PARTITION:-ex01A800}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_MEM="${SLURM_MEM:-80G}"
SLURM_TIME="${SLURM_TIME:-01:00:00}"
CONDA_ENV="${JEPA_CONDA_ENV-newbase}"      # Empty uses the already active Python environment.
CONDA_HOME="${JEPA_CONDA_HOME:-/share/software/anaconda3}"

DRY_RUN=0
LOCAL=0
SOURCE_ARGUMENT=0
for argument in "$@"; do
    case "$argument" in
        --dry-run) DRY_RUN=1 ;;
        --local) LOCAL=1 ;;
        -h|--help)
            echo 'Usage: bash final_exam/single_circuit_pin/inspect_pe_attention_pin.sh [RESULT_PATH] [--dry-run] [--local]'
            echo 'Edit SOURCE_RUN, CIRCUIT, N_SAMPLES, MODELS, SEED and SPLIT at the top.'
            echo 'Default: submit read-only PE attention inspection; no training or gate disabling. --local runs in the current allocation.'
            echo '--dry-run prints commands only and creates no directories.'
            exit 0 ;;
        --*) echo "Unknown option: $argument" >&2; exit 2 ;;
        *)
            (( SOURCE_ARGUMENT == 0 )) || { echo 'Supply only one result path.' >&2; exit 2; }
            SOURCE_RUN="$argument"; SOURCE_ARGUMENT=1 ;;
    esac
done
[[ -n "$SOURCE_RUN" ]] || { echo 'Set SOURCE_RUN at the top or pass the result path as an argument.' >&2; exit 2; }
[[ "$CIRCUIT" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'Invalid CIRCUIT.' >&2; exit 2; }
for count in "$N_SAMPLES" "$BATCH_SIZE" "$VIEWER_SAMPLES" "$SLURM_CPUS"; do
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo 'Counts and CPUs must be positive integers.' >&2; exit 2; }
done
[[ "$SEED" =~ ^[0-9]+$ && "$SAMPLE_SEED" =~ ^[0-9]+$ && "$SLURM_GPUS" =~ ^[0-9]+$ ]] || {
    echo 'Seeds and GPU count must be nonnegative integers.' >&2; exit 2;
}
(( ${#MODELS[@]} )) || { echo 'MODELS must be nonempty.' >&2; exit 2; }
if [[ "$DEVICE" == cuda* ]] && (( ! LOCAL && SLURM_GPUS < 1 )); then
    echo 'CUDA inspection requires a requested GPU.' >&2; exit 2
fi
cd "$PROJECT_ROOT"
export CIRCUIT_JEPA_ROOT="$PROJECT_ROOT" JEPA_PROJECT_ROOT="$PROJECT_ROOT"
export JEPA_CONDA_ENV="$CONDA_ENV" JEPA_CONDA_HOME="$CONDA_HOME" PYTHONUNBUFFERED=1
run_id="${CIRCUIT}_pe_attention_probe_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}_pin"
result_dir="$HERE/results_pin/$run_id"
log_dir="$HERE/logs_pin/$run_id"
hpc_dir="$HERE/hpc_logs_pin/$run_id"
arguments=(--source-run "$SOURCE_RUN" --circuit "$CIRCUIT" --num-samples "$N_SAMPLES"
    --models "${MODELS[@]}" --seed "$SEED" --split "$SPLIT" --sampling-seed "$SAMPLE_SEED"
    --batch-size "$BATCH_SIZE" --viewer-samples "$VIEWER_SAMPLES" --device "$DEVICE"
    --output "$result_dir" --log-dir "$log_dir")
runner=(bash "$PROJECT_ROOT/scripts/run/slurm_run.sh"
    "${JEPA_PYTHON:-python}" -u "$PROJECT_ROOT/final_exam/attention_probe_pin.py" "${arguments[@]}")
if (( LOCAL )); then
    command=("${runner[@]}")
else
    options=(--parsable --job-name="pe-attention-${CIRCUIT//_/-}-pin" --chdir="$PROJECT_ROOT"
        --output="$hpc_dir/%j_pin.out" --error="$hpc_dir/%j_pin.err"
        --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS"
        --mem="$SLURM_MEM" --time="$SLURM_TIME" --export=ALL)
    if (( SLURM_GPUS > 0 )); then options+=(--gres="gpu:$SLURM_GPUS"); fi
    if [[ -n "$SLURM_ACCOUNT" ]]; then options+=(--account="$SLURM_ACCOUNT"); fi
    if [[ -n "$SLURM_PARTITION" ]]; then options+=(--partition="$SLURM_PARTITION"); fi
    command=(sbatch "${options[@]}" "$PROJECT_ROOT/scripts/run/slurm_run.sh"
        "${JEPA_PYTHON:-python}" -u "$PROJECT_ROOT/final_exam/attention_probe_pin.py" "${arguments[@]}")
fi
printf '%q ' "${command[@]}"
printf '\n'
if (( DRY_RUN )); then exit 0; fi
# This uses only stdlib/metadata, so it is safe on a login node without torch.
"${JEPA_INSPECT_PYTHON:-python3}" "$PROJECT_ROOT/final_exam/attention_probe_pin.py" "${arguments[@]}" --inspect-source
if (( LOCAL )); then exec "${command[@]}"; fi
command -v sbatch >/dev/null || { echo 'sbatch is unavailable; use --local inside a suitable environment.' >&2; exit 1; }
mkdir -p "$HERE/hpc_logs_pin"
mkdir "$hpc_dir"
printf '%q ' "${command[@]}" > "$hpc_dir/submission_pin.txt"
printf '\n' >> "$hpc_dir/submission_pin.txt"
job_id="$("${command[@]}")"
printf 'Submitted PE attention probe job %s\nResults: %s\nExperiment log: %s\nHPC logs: %s\n' \
    "$job_id" "$result_dir" "$log_dir" "$hpc_dir"
