#!/usr/bin/env bash
# Absolute snapshot root is passed by the submitter; never derive it from Slurm spool.
set -euo pipefail
export CIRCUIT_JEPA_ROOT="$1" JEPA_PROJECT_ROOT="$1"
export JEPA_CONDA_HOME="$2" JEPA_CONDA_ENV="$3" JEPA_CODE_COMMIT="$4"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
shift 4
exec bash "$CIRCUIT_JEPA_ROOT/scripts/run/slurm_run.sh" "${JEPA_PYTHON:-python}" -u \
    -m final_exam.multi_circuit.worker_pin "$@"
