#!/usr/bin/env bash
# Same Slurm allocation and output locations as the paired branch probe.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    echo 'Usage: bash experiments/branch_probe/submit_diagnostics.sh [--dry-run | --submit] --data FILE --pretrained FILE [diagnose.py options]'
    echo 'Default: preview only. Add --submit to submit one diagnostic GPU job.'
    exit 0
fi
exec bash "$HERE/submit.sh" --diagnose "$@"
