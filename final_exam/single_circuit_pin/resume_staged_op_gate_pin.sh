#!/usr/bin/env bash
# Resume saved staged-gate progress into a new in-project run directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"

# Set this once, or pass the old run directory as the first argument.
# The original run's models, seeds, config and split are reused unchanged.
RESUME_FROM="${RESUME_FROM:-}"
DEVICE="cuda"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-b_phzhwu}"
SLURM_PARTITION="${SLURM_PARTITION:-ex01A800}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_MEM="${SLURM_MEM:-80G}"
SLURM_TIME="${SLURM_TIME:-10-00:00:00}"
CONDA_ENV="${JEPA_CONDA_ENV:-newbase}"
CONDA_HOME="${JEPA_CONDA_HOME:-/share/software/anaconda3}"
SLURM_MAIL_USER="${SLURM_MAIL_USER:-}"
SLURM_MAIL_TYPE="${SLURM_MAIL_TYPE:-BEGIN,END,FAIL}"
PYTHON="${JEPA_PYTHON:-python}"

DRY_RUN=0
source_argument=0
for argument in "$@"; do
    case "$argument" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            echo 'Usage: bash final_exam/single_circuit_pin/resume_staged_op_gate_pin.sh [OLD_RUN_OR_TRAINING_DIR] [--dry-run]'
            echo 'Alternatively edit RESUME_FROM above. Original models, seeds, config and split are retained.'
            echo 'Creates a fresh result/log directory, reuses canonical JSON, and never modifies the source run.'
            exit 0 ;;
        --*) echo "Unknown option: $argument; use --help" >&2; exit 2 ;;
        *)
            (( source_argument == 0 )) || { echo 'Specify only one source directory.' >&2; exit 2; }
            RESUME_FROM="$argument"
            source_argument=1 ;;
    esac
done
[[ -n "$RESUME_FROM" ]] || { echo 'Set RESUME_FROM or pass the old run directory; use --help.' >&2; exit 2; }
[[ "$SLURM_CPUS" =~ ^[1-9][0-9]*$ && "$SLURM_GPUS" =~ ^[0-9]+$ ]] || {
    echo 'Invalid CPU/GPU request.' >&2; exit 2;
}
if [[ "$DEVICE" == cuda* ]] && (( SLURM_GPUS < 1 )); then
    echo 'CUDA training requires at least one requested GPU.' >&2
    exit 2
fi
cd "$PROJECT_ROOT"
# Inspection is stdlib-only: no raw data import, torch import, CUDA allocation,
# or bytecode writes on the login node (including --dry-run).
source_metadata="$("$PYTHON" -B "$PROJECT_ROOT/final_exam/common/resume_pin.py" --resume-from "$RESUME_FROM" --inspect)"
circuit="$("$PYTHON" -B -c 'import json,sys; print(json.load(sys.stdin)["circuit"])' <<< "$source_metadata")"
source_training="$("$PYTHON" -B -c 'import json,sys; print(json.load(sys.stdin)["source_training_output"])' <<< "$source_metadata")"
if (( ! DRY_RUN )); then
    command -v sbatch >/dev/null || { echo 'sbatch is unavailable.' >&2; exit 1; }
    source_job="$("$PYTHON" -B -c 'import json,sys; print(json.load(sys.stdin).get("source_slurm_job_id") or "")' <<< "$source_metadata")"
    if [[ "$source_job" =~ ^[0-9]+$ ]] && command -v squeue >/dev/null 2>&1; then
        # Do not create a second training job while the original is still live.
        # Missing/expired Slurm records are normal after an older failed run.
        if active_states="$(squeue --noheader --jobs="$source_job" --format='%T' 2>/dev/null)"; then
            if [[ -n "${active_states//[[:space:]]/}" ]]; then
                printf 'Source Slurm job %s is still active (%s). Stop it before resuming.\n' \
                    "$source_job" "$active_states" >&2
                exit 2
            fi
        fi
    fi
fi

export JEPA_PROJECT_ROOT="$PROJECT_ROOT" CIRCUIT_JEPA_ROOT="$PROJECT_ROOT"
export JEPA_CONDA_ENV="$CONDA_ENV" JEPA_CONDA_HOME="$CONDA_HOME"
export PYTHONUNBUFFERED=1
run_id="${circuit}_resume_staged_op_gate_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}_pin"
hpc_dir="$HERE/hpc_logs_pin/$run_id"
log_dir="$HERE/logs_pin/$run_id"
result_dir="$HERE/results_pin/$run_id"
options=(
    --parsable --job-name="final-resume-gate-${circuit//_/-}-pin" --chdir="$PROJECT_ROOT"
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
# slurm_run.sh locates the repository through the exported root, not its spool copy.
command=(sbatch "${options[@]}" "$PROJECT_ROOT/scripts/run/slurm_run.sh"
    "$PYTHON" -u "$PROJECT_ROOT/final_exam/common/resume_pin.py"
    --resume-from "$source_training" --run-id "$run_id" --device "$DEVICE"
    --submitted-source "$hpc_dir/submitted_resume_source_pin.json")
printf '%q ' "${command[@]}"
printf '\n'
if (( DRY_RUN )); then exit 0; fi
mkdir -p "$HERE/hpc_logs_pin"
mkdir "$hpc_dir"
printf '%s\n' "$source_metadata" > "$hpc_dir/submitted_resume_source_pin.json"
if job_id="$("${command[@]}")"; then
    printf 'Submitted resume job %s\nSource: %s\nHPC logs: %s\nExperiment logs: %s\nResults: %s\n' \
        "$job_id" "$source_training" "$hpc_dir" "$log_dir" "$result_dir"
    printf 'job_id=%s\nrun_id=%s\nresume_from=%s\n' \
        "$job_id" "$run_id" "$source_training" > "$hpc_dir/submission_pin.txt"
else
    status=$?
    printf 'Submission failed with exit code %s. HPC directory: %s\n' "$status" "$hpc_dir" >&2
    exit "$status"
fi
