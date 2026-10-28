#!/usr/bin/env bash
# Preview by default. All diagnostic logs/results stay inside branch_probe.
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
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --submit) DRY_RUN=0 ;;
        -h|--help)
            echo 'Usage: bash experiments/branch_probe/submit.sh [--dry-run | --submit] --data FILE --pretrained PRETRAIN_BEST [run.py options]'
            echo 'Default: preview only. --submit or DRY_RUN=0 submits one job.'
            echo 'Resource/account/environment settings may be overridden through environment variables.'
            exit 0 ;;
        *) ARGS+=("$arg") ;;
    esac
done
[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo 'DRY_RUN must be 0 or 1' >&2; exit 2; }
[[ "$SLURM_CPUS" =~ ^[1-9][0-9]*$ && "$SLURM_GPUS" =~ ^[0-9]+$ ]] || { echo 'Invalid CPU/GPU count' >&2; exit 2; }
(( ${#ARGS[@]} > 0 )) || { echo 'Supply --data and --pretrained; use --help for usage.' >&2; exit 2; }
[[ ! -L "$HERE/logs" ]] || { echo 'branch_probe/logs must not be a symlink outside the experiment folder' >&2; exit 2; }
LOG_DIR="$HERE/logs/$(date -u +%Y%m%dT%H%M%SZ)-$$"
OPTIONS=(--parsable --job-name=circuit-jepa-branch-probe --chdir="$PROJECT_ROOT"
    --output="$LOG_DIR/%j.out" --error="$LOG_DIR/%j.err" --nodes=1 --ntasks=1
    --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS" --mem="$SLURM_MEM"
    --time="$SLURM_TIME" --export=ALL)
if (( SLURM_GPUS > 0 )); then OPTIONS+=(--gres="gpu:$SLURM_GPUS"); fi
if [[ -n "$SLURM_ACCOUNT" ]]; then OPTIONS+=(--account="$SLURM_ACCOUNT"); fi
if [[ -n "$SLURM_PARTITION" ]]; then OPTIONS+=(--partition="$SLURM_PARTITION"); fi
COMMAND=(sbatch "${OPTIONS[@]}" "$PROJECT_ROOT/scripts/run/slurm_run.sh"
    "${JEPA_PYTHON:-python}" "$HERE/run.py" "${ARGS[@]}")
printf '%q ' "${COMMAND[@]}"; printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then exit 0; fi
command -v sbatch >/dev/null || { echo 'sbatch unavailable; run run.py directly on an allocated compute node.' >&2; exit 1; }
mkdir -p "$LOG_DIR"
"${COMMAND[@]}"
