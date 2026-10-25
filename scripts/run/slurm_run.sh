#!/usr/bin/env bash
# Resource/account directives are supplied by submit_comparison.sh, not duplicated here.
set -euo pipefail
PROJECT_ROOT="${CIRCUIT_JEPA_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
[[ -f "$PROJECT_ROOT/device_jepa/cli.py" ]] || { echo 'Invalid Circuit-OP-JEPA root' >&2; exit 1; }
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${JEPA_CONDA_ENV:-}" ]]; then
    CONDA_BASE="${JEPA_CONDA_HOME:-/share/software/anaconda3}"
    if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] && command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    fi
    [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] || { echo 'Set JEPA_CONDA_HOME to a valid Conda installation' >&2; exit 1; }
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$JEPA_CONDA_ENV"
fi
printf 'Job %s | Node %s | Command:' "${SLURM_JOB_ID:-local}" "${SLURM_NODELIST:-local}"
printf ' %q' "$@"; printf '\n'
exec "$@"
