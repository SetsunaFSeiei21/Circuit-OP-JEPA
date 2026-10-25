#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export JEPA_JOB_KIND=smoke_test
export SLURM_TIME="${SLURM_TIME:-00:30:00}"
exec bash "$PROJECT_ROOT/scripts/run/submit_comparison.sh" "$@" --smoke
