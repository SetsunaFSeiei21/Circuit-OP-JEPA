#!/usr/bin/env bash
set -euo pipefail
# Mirrors Analog-Agent scripts/run layout/resources. Override via environment.
# Safe by default: DRY_RUN=0 explicitly submits. No cluster jobs run in tests.
DRY_RUN="${DRY_RUN:-1}"
JEPA_RAW_INPUT="${JEPA_RAW_INPUT:-0}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_MEM="${SLURM_MEM:-80G}"
SLURM_TIME="${SLURM_TIME:-10-00:00:00}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_MAIL_USER="${SLURM_MAIL_USER:-}"
SLURM_MAIL_TYPE="${SLURM_MAIL_TYPE:-BEGIN,END,FAIL}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CIRCUIT_JEPA_ROOT="$PROJECT_ROOT"
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo 'DRY_RUN must be 0 or 1' >&2; exit 2; }
[[ "$JEPA_RAW_INPUT" =~ ^[01]$ ]] || { echo 'JEPA_RAW_INPUT must be 0 or 1' >&2; exit 2; }
[[ "$SLURM_CPUS" =~ ^[1-9][0-9]*$ && "$SLURM_GPUS" =~ ^[0-9]+$ ]] || { echo 'Invalid CPU/GPU count' >&2; exit 2; }
(( $# > 0 )) || { echo 'Usage: submit_comparison.sh --data FILE --config FILE --output DIR [--seeds 17 29 43] [--device cuda]' >&2; exit 2; }
LOG_DIR="$PROJECT_ROOT/hpc_logs/${JEPA_JOB_KIND:-run}/$(date -u +%Y%m%dT%H%M%SZ)-$$"
OPTIONS=(--parsable --job-name="circuit-jepa-${JEPA_JOB_KIND:-run}" --chdir="$PROJECT_ROOT"
    --output="$LOG_DIR/%j.out" --error="$LOG_DIR/%j.err" --nodes=1 --ntasks=1
    --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS" --mem="$SLURM_MEM" --time="$SLURM_TIME" --export=ALL)
if (( SLURM_GPUS > 0 )); then OPTIONS+=(--gres="gpu:$SLURM_GPUS"); fi
if [[ -n "$SLURM_ACCOUNT" ]]; then OPTIONS+=(--account="$SLURM_ACCOUNT"); fi
if [[ -n "$SLURM_PARTITION" ]]; then OPTIONS+=(--partition="$SLURM_PARTITION"); fi
if [[ -n "$SLURM_MAIL_USER" ]]; then OPTIONS+=(--mail-user="$SLURM_MAIL_USER" --mail-type="$SLURM_MAIL_TYPE"); fi
COMMAND=(sbatch "${OPTIONS[@]}" "$PROJECT_ROOT/scripts/run/slurm_run.sh" bash "$PROJECT_ROOT/scripts/run/run_comparison.sh" "$@")
if [[ "$JEPA_RAW_INPUT" == 1 ]]; then
    COMMAND=(sbatch "${OPTIONS[@]}" "$PROJECT_ROOT/scripts/run/slurm_run.sh" "${JEPA_PYTHON:-python}" "$PROJECT_ROOT/scripts/run/run_raw_comparison.py" "$@")
fi
printf '%q ' "${COMMAND[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi
command -v sbatch >/dev/null || { echo 'sbatch unavailable; use local run_comparison.sh' >&2; exit 1; }
mkdir -p "$LOG_DIR"
"${COMMAND[@]}"
