#!/usr/bin/env bash
# Example: bash final_exam/single_circuit_pin/summarize_pin.sh path/to/run --output new_pin.xlsx
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
if (( $# == 0 )); then
    echo 'Usage: bash final_exam/single_circuit_pin/summarize_pin.sh RUN_DIRECTORY [--output FILE_pin.xlsx] [--recover-export]' >&2
    exit 2
fi
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${JEPA_PYTHON:-python}" "$PROJECT_ROOT/final_exam/summarize_pin.py" "$@"
