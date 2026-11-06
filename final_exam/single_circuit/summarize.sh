#!/usr/bin/env bash
# Example: bash final_exam/single_circuit/summarize.sh path/to/run --output new.xlsx
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
if (( $# == 0 )); then
    echo 'Usage: bash final_exam/single_circuit/summarize.sh RUN_DIRECTORY [--output FILE.xlsx]' >&2
    exit 2
fi
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${JEPA_PYTHON:-python}" "$PROJECT_ROOT/final_exam/summarize.py" "$@"
