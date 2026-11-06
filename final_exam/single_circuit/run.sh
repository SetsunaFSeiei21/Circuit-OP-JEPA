#!/usr/bin/env bash
# Slurm executes this entrypoint on the allocated compute node.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
if [[ -n "${JEPA_CONDA_ENV:-}" ]]; then
    conda_base="${JEPA_CONDA_HOME:-/share/software/anaconda3}"
    if [[ ! -f "$conda_base/etc/profile.d/conda.sh" ]] && command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base)"
    fi
    [[ -f "$conda_base/etc/profile.d/conda.sh" ]] || {
        echo 'Set JEPA_CONDA_HOME to a valid Conda installation.' >&2; exit 1;
    }
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$JEPA_CONDA_ENV"
fi
printf 'Job %s | Node %s | Python %s\n' \
    "${SLURM_JOB_ID:-local}" "${SLURM_JOB_NODELIST:-local}" "$(command -v "${JEPA_PYTHON:-python}")"
exec "${JEPA_PYTHON:-python}" -u "$PROJECT_ROOT/final_exam/common/launch.py" "$@"
