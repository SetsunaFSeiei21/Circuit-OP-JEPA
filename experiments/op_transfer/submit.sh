#!/usr/bin/env bash
# Preview by default; logs and results stay under this experiment directory.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
export CIRCUIT_JEPA_ROOT="$PROJECT_ROOT"

DRY_RUN="${DRY_RUN:-1}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_MEM="${SLURM_MEM:-80G}"
SLURM_TIME="${SLURM_TIME:-10-00:00:00}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-b_phzhwu}"
SLURM_PARTITION="${SLURM_PARTITION:-ex01A800}"
export JEPA_CONDA_ENV="${JEPA_CONDA_ENV-newbase}"
export JEPA_CONDA_HOME="${JEPA_CONDA_HOME:-/share/software/anaconda3}"

ARGS=()
ENTRY="$HERE/run.py"
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --submit) DRY_RUN=0 ;;
        --list-models) exec "${JEPA_PYTHON:-python}" "$HERE/run.py" --list-models ;;
        -h|--help)
            echo 'Usage: bash experiments/op_transfer/submit.sh [--dry-run | --submit] --circuit NAME --models MODEL...'
            echo 'Raw OOD: --train-circuits NAME... --ood-circuits NAME...; canonical data: --data FILE'
            echo 'Default: preview only. --submit or DRY_RUN=0 submits one Slurm job.'
            echo 'Use run.py --help for experiment arguments; --list-models lists selectable names.'
            echo 'Resource/account/environment settings may be overridden through environment variables.'
            exit 0 ;;
        --circuit|--circuit=*|--train-circuits|--train-circuits=*) ENTRY="$HERE/run_raw.py"; ARGS+=("$arg") ;;
        *) ARGS+=("$arg") ;;
    esac
done
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo 'DRY_RUN must be 0 or 1' >&2; exit 2; }
[[ "$SLURM_CPUS" =~ ^[1-9][0-9]*$ && "$SLURM_GPUS" =~ ^[0-9]+$ ]] || { echo 'Invalid CPU/GPU count' >&2; exit 2; }
(( ${#ARGS[@]} > 0 )) || { echo 'Supply --circuit or --data; use --help for usage.' >&2; exit 2; }
[[ ! -L "$HERE/logs" ]] || { echo 'op_transfer/logs must not be a symlink' >&2; exit 2; }
LOG_DIR="$HERE/logs/$(date -u +%Y%m%dT%H%M%SZ)-$$"
OPTIONS=(--parsable --job-name=circuit-jepa-op-transfer --chdir="$PROJECT_ROOT"
    --output="$LOG_DIR/%j.out" --error="$LOG_DIR/%j.err" --nodes=1 --ntasks=1
    --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS" --mem="$SLURM_MEM"
    --time="$SLURM_TIME" --export=ALL)
if (( SLURM_GPUS > 0 )); then OPTIONS+=(--gres="gpu:$SLURM_GPUS"); fi
if [[ -n "$SLURM_ACCOUNT" ]]; then OPTIONS+=(--account="$SLURM_ACCOUNT"); fi
if [[ -n "$SLURM_PARTITION" ]]; then OPTIONS+=(--partition="$SLURM_PARTITION"); fi
COMMAND=(sbatch "${OPTIONS[@]}" "$PROJECT_ROOT/scripts/run/slurm_run.sh"
    "${JEPA_PYTHON:-python}" "$ENTRY" "${ARGS[@]}")
printf '%q ' "${COMMAND[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi
command -v sbatch >/dev/null || { echo 'sbatch unavailable; run run.py directly on an allocated compute node.' >&2; exit 1; }
mkdir -p "$LOG_DIR"
"${COMMAND[@]}"
