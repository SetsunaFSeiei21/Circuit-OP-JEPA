#!/usr/bin/env bash
# Edit settings below. Analyze each checkpoint independently; never pool circuit weights.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"

# CIRCUITS and SOURCE_RUNS must match by position. A result root or training_pin works.
CIRCUITS=(Opamp1 two_stage_folded_opamp)
SOURCE_RUNS=(
    ""  # Opamp1: /share/.../results_pin/Opamp1_width_decoder_..._pin
    ""  # two_stage_folded_opamp: /share/.../results_pin/two_stage_folded_opamp_width_decoder_..._pin
)
MODELS=(width_decoder_pin)          # A saved model with one decoder branch per PE.
SEEDS=(17)                         # Saved checkpoint seeds; use (17 29 43) when all exist.
RANKS=(8 16 32 64)
N_SAMPLES=128                      # Train samples for bounded activation diagnostics; 0 disables.
N_VAL_SAMPLES=128                  # Validation samples for temporary reconstructed models; 0 disables.
MAX_TOKENS=512                     # Bounded activation vectors per matrix across all selected samples.
SAMPLE_SEED=17
BATCH_SIZE=16
SCOPE="all"                       # all / attention / ffn; corresponding decoder projections only.
COMPONENT="performance"           # performance / op_predictor / both; OP side needs wide OP decoder.
VALIDATION_SCOPE="all"             # all / group / both; selected groups jointly or one group at a time.
MAX_PLOTS=12                      # Per-checkpoint group plots; 0 requires no matplotlib.
DEVICE="${DEVICE:-cuda}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-b_phzhwu}"
SLURM_PARTITION="${SLURM_PARTITION:-ex01A800}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_MEM="${SLURM_MEM:-80G}"
SLURM_TIME="${SLURM_TIME:-02:00:00}"
CONDA_ENV="${JEPA_CONDA_ENV-newbase}"
CONDA_HOME="${JEPA_CONDA_HOME:-/share/software/anaconda3}"
DRY_RUN=0
LOCAL=0
for argument in "$@"; do
    case "$argument" in
        --dry-run) DRY_RUN=1 ;;
        --local) LOCAL=1 ;;
        -h|--help)
            echo 'Usage: bash final_exam/single_circuit_pin/inspect_shared_decoder_pin.sh [--dry-run] [--local]'
            echo 'Edit CIRCUITS, SOURCE_RUNS, MODELS, SEEDS, RANKS and N_SAMPLES at the top.'
            echo 'Default submits one read-only Slurm GPU diagnostic job; --local uses the current allocation.'
            echo 'Shared down basis is fitted within each checkpoint across its PE decoders; circuits never share fits.'
            echo 'No training; train activation capture and optional validation only; source checkpoints are untouched.'
            exit 0 ;;
        *) echo "Unknown option: $argument" >&2; exit 2 ;;
    esac
done
(( ${#CIRCUITS[@]} && ${#CIRCUITS[@]} == ${#SOURCE_RUNS[@]} && ${#MODELS[@]} && ${#SEEDS[@]} && ${#RANKS[@]} )) || {
    echo 'CIRCUITS/SOURCE_RUNS must be nonempty and paired; MODELS/SEEDS/RANKS must be nonempty.' >&2; exit 2;
}
for count in "$N_SAMPLES" "$N_VAL_SAMPLES" "$MAX_PLOTS" "$SAMPLE_SEED" "$SLURM_GPUS" "${SEEDS[@]}"; do
    [[ "$count" =~ ^[0-9]+$ ]] || { echo 'Counts/seeds must be nonnegative integers.' >&2; exit 2; }
done
for count in "$MAX_TOKENS" "$BATCH_SIZE" "$SLURM_CPUS" "${RANKS[@]}"; do
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo 'Ranks/token cap/batch size/CPUs must be positive.' >&2; exit 2; }
done
if [[ "$DEVICE" == cuda* ]] && (( ! LOCAL && SLURM_GPUS < 1 )); then
    echo 'CUDA diagnostics require at least one requested GPU.' >&2; exit 2
fi
case "$SCOPE" in all|attention|ffn) ;; *) echo "Invalid SCOPE: $SCOPE" >&2; exit 2 ;; esac
case "$COMPONENT" in performance|op_predictor|both) ;; *) echo "Invalid COMPONENT: $COMPONENT" >&2; exit 2 ;; esac
case "$VALIDATION_SCOPE" in all|group|both) ;; *) echo "Invalid VALIDATION_SCOPE: $VALIDATION_SCOPE" >&2; exit 2 ;; esac
cd "$PROJECT_ROOT"
export CIRCUIT_JEPA_ROOT="$PROJECT_ROOT" JEPA_PROJECT_ROOT="$PROJECT_ROOT"
export JEPA_CONDA_ENV="$CONDA_ENV" JEPA_CONDA_HOME="$CONDA_HOME" PYTHONUNBUFFERED=1
run_id="shared_decoder_probe_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}_pin"
result_dir="$HERE/results_pin/$run_id"
log_dir="$HERE/logs_pin/$run_id"
hpc_dir="$HERE/hpc_logs_pin/$run_id"
arguments=()
for i in "${!CIRCUITS[@]}"; do
    [[ -n "${SOURCE_RUNS[$i]}" ]] || { echo "Fill SOURCE_RUNS[$i] for ${CIRCUITS[$i]}." >&2; exit 2; }
    arguments+=(--source "${CIRCUITS[$i]}=${SOURCE_RUNS[$i]}")
done
arguments+=(--models "${MODELS[@]}" --seeds "${SEEDS[@]}" --ranks "${RANKS[@]}"
    --num-samples "$N_SAMPLES" --num-val-samples "$N_VAL_SAMPLES" --max-tokens "$MAX_TOKENS" --sampling-seed "$SAMPLE_SEED"
    --batch-size "$BATCH_SIZE" --scope "$SCOPE" --component "$COMPONENT" --max-plots "$MAX_PLOTS"
    --validation-scope "$VALIDATION_SCOPE"
    --device "$DEVICE" --output "$result_dir" --log-dir "$log_dir")
runner=("$PROJECT_ROOT/scripts/run/slurm_run.sh" "${JEPA_PYTHON:-python}" -u
    "$PROJECT_ROOT/final_exam/shared_decoder_probe_pin.py" "${arguments[@]}")
if (( LOCAL )); then
    command=(bash "${runner[@]}")
else
    options=(--parsable --job-name=shared-decoder-pin --chdir="$PROJECT_ROOT"
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
"${JEPA_INSPECT_PYTHON:-python3}" "$PROJECT_ROOT/final_exam/shared_decoder_probe_pin.py" "${arguments[@]}" --inspect-source
if (( LOCAL )); then exec "${command[@]}"; fi
command -v sbatch >/dev/null || { echo 'sbatch unavailable; use --local in a compute allocation.' >&2; exit 1; }
mkdir -p "$HERE/hpc_logs_pin"
mkdir "$hpc_dir"
cp -- "$HERE/inspect_shared_decoder_pin.sh" "$hpc_dir/submitted_inspect_shared_decoder_pin.sh"
printf '%q ' "${command[@]}" > "$hpc_dir/submission_pin.txt"
printf '\n' >> "$hpc_dir/submission_pin.txt"
job_id="$("${command[@]}")"
printf 'Submitted shared decoder probe job %s\nResults: %s\nExperiment log: %s\nHPC logs: %s\n' \
    "$job_id" "$result_dir" "$log_dir" "$hpc_dir"
