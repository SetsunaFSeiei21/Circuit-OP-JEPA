#!/usr/bin/env bash
# Slurm executes this entrypoint on the allocated compute node.
set -euo pipefail
# Slurm runs a spool copy of this file, so BASH_SOURCE is not the repository.
# The submitter exports its root; --chdir and SLURM_SUBMIT_DIR cover direct sbatch use.
if [[ -n "${JEPA_PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT="$JEPA_PROJECT_ROOT"
elif [[ -f "$PWD/final_exam/common/launch.py" ]]; then
    PROJECT_ROOT="$PWD"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/final_exam/common/launch.py" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
fi
[[ -f "$PROJECT_ROOT/final_exam/common/launch.py" ]] || {
    printf 'Cannot locate final_exam project root: %s\nSet JEPA_PROJECT_ROOT to the repository directory.\n' "$PROJECT_ROOT" >&2
    exit 2
}
cd -- "$PROJECT_ROOT"
PROJECT_ROOT="$PWD"
export JEPA_PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
printf 'Project root: %s\n' "$PROJECT_ROOT"
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
