#!/usr/bin/env bash
set -euo pipefail
# Same cluster allocation/Conda handling as canonical experiments. Import is
# performed on the allocated compute node, never by this submission wrapper.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export JEPA_RAW_INPUT=1
export JEPA_JOB_KIND="${JEPA_JOB_KIND:-raw_run}"
exec bash "$PROJECT_ROOT/scripts/run/submit_comparison.sh" "$@"
