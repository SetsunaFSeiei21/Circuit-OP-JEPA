#!/usr/bin/env bash
# Edit SOURCE_RUNS; sources are read-only. No training or new sample selection.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"

# Result roots (containing training_pin), or training_pin directories themselves.
# Several runs of the same circuit are allowed if they have different models/seeds.
SOURCE_RUNS=(
    ""  # /share/.../results_pin/two_stage_opamp_otaf_lasd_moe_..._pin
    # "/public/.../results_pin/Opamp1_lasd_moe_..._pin"
)
MODELS=()                           # Empty: all completed performance models.
SEEDS=()                            # Empty: all completed seeds; no best/worst selection.
BASELINE="device_op_cascade_pin"
SPLITS=(val test)
BATCH_SIZE=32
BOOTSTRAP=2000                      # Paired training-seed bootstrap; 0 disables CI only.
ANALYSIS_SEED=17
TOP_ERRORS=20                       # Export largest residuals and paired improvements/degradations.
MAX_PLOT_POINTS=5000                # Display limit only; all saved rows are scored.
PLOTS=1
DEVICE="${DEVICE:-cuda}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-b_phzhwu}"
SLURM_PARTITION="${SLURM_PARTITION:-ex01A800}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_MEM="${SLURM_MEM:-32G}"
SLURM_TIME="${SLURM_TIME:-04:00:00}"
export JEPA_CONDA_ENV="${JEPA_CONDA_ENV-newbase}"
export JEPA_CONDA_HOME="${JEPA_CONDA_HOME:-/share/software/anaconda3}"
DRY_RUN=0
LOCAL=0
for argument in "$@"; do
    case "$argument" in
        --dry-run) DRY_RUN=1 ;;
        --local) LOCAL=1 ;;
        -h|--help)
            echo 'Usage: bash final_exam/single_circuit_pin/inspect_predictions_pin.sh [--dry-run] [--local]'
            echo 'Set SOURCE_RUNS; MODELS/SEEDS empty means all completed model/seed checkpoints.'
            echo 'Default: one Slurm job, sequential read-only inference; --local uses current allocation.'
            echo 'Replays complete saved val/test cohorts. Plot point limits never change statistics.'
            echo 'No training, normalization fitting, source writes, or best/worst seed selection.'
            exit 0 ;;
        *) echo "Unknown option: $argument" >&2; exit 2 ;;
    esac
done
(( ${#SOURCE_RUNS[@]} )) || { echo 'Fill SOURCE_RUNS.' >&2; exit 2; }
for source in "${SOURCE_RUNS[@]}"; do
    [[ -n "$source" ]] || { echo 'Fill every SOURCE_RUNS entry.' >&2; exit 2; }
done
for count in "$BATCH_SIZE" "$MAX_PLOT_POINTS" "$SLURM_CPUS"; do
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo 'Batch size/plot limit/CPUs must be positive.' >&2; exit 2; }
done
for count in "$BOOTSTRAP" "$ANALYSIS_SEED" "$TOP_ERRORS" "$SLURM_GPUS" "${SEEDS[@]}"; do
    [[ "$count" =~ ^[0-9]+$ ]] || { echo 'Counts/seeds must be nonnegative integers.' >&2; exit 2; }
done
[[ "$PLOTS" == 0 || "$PLOTS" == 1 ]] || { echo 'PLOTS must be 0 or 1.' >&2; exit 2; }
if [[ "$DEVICE" == cuda* ]] && (( ! LOCAL && SLURM_GPUS < 1 )); then
    echo 'CUDA diagnostics require at least one GPU.' >&2; exit 2
fi
cd "$PROJECT_ROOT"
export JEPA_PROJECT_ROOT="$PROJECT_ROOT" PYTHONUNBUFFERED=1
run_id="prediction_probe_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}_pin"
result_dir="$HERE/results_pin/$run_id"
log_dir="$HERE/logs_pin/$run_id"
hpc_dir="$HERE/hpc_logs_pin/$run_id"
arguments=(--source-runs "${SOURCE_RUNS[@]}" --baseline "$BASELINE" --splits "${SPLITS[@]}"
    --batch-size "$BATCH_SIZE" --bootstrap "$BOOTSTRAP" --analysis-seed "$ANALYSIS_SEED"
    --top-errors "$TOP_ERRORS" --max-plot-points "$MAX_PLOT_POINTS" --device "$DEVICE"
    --output "$result_dir" --log-dir "$log_dir")
if (( ${#MODELS[@]} )); then arguments+=(--models "${MODELS[@]}"); fi
if (( ${#SEEDS[@]} )); then arguments+=(--seeds "${SEEDS[@]}"); fi
if (( ! PLOTS )); then arguments+=(--no-plots); fi
runner=("$PROJECT_ROOT/scripts/run/slurm_run.sh" "${JEPA_PYTHON:-python}" -u
    "$PROJECT_ROOT/final_exam/prediction_probe_pin.py" "${arguments[@]}")
if (( LOCAL )); then
    command=(bash "${runner[@]}")
else
    options=(--parsable --job-name=prediction-probe-pin --chdir="$PROJECT_ROOT"
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
"${JEPA_INSPECT_PYTHON:-python3}" "$PROJECT_ROOT/final_exam/prediction_probe_pin.py" "${arguments[@]}" --inspect-source
if (( LOCAL )); then exec "${command[@]}"; fi
command -v sbatch >/dev/null || { echo 'sbatch unavailable; use --local inside a compute allocation.' >&2; exit 1; }
mkdir -p "$HERE/hpc_logs_pin"
mkdir "$hpc_dir"
cp -- "$HERE/inspect_predictions_pin.sh" "$hpc_dir/submitted_inspect_predictions_pin.sh"
printf '%q ' "${command[@]}" > "$hpc_dir/submission_pin.txt"
printf '\n' >> "$hpc_dir/submission_pin.txt"
job_id="$("${command[@]}")"
printf 'Submitted prediction probe job %s\nResults: %s\nExperiment logs: %s\nHPC logs: %s\n' \
    "$job_id" "$result_dir" "$log_dir" "$hpc_dir"
