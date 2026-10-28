#!/usr/bin/env bash
# Submit the controlled readout experiment using the existing branch-probe allocation.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    echo 'Usage: bash experiments/branch_probe/submit_ablation.sh [--dry-run | --submit] --data FILE --pretrained FILE [ablate.py options]'
    echo 'Default: preview only. Add --submit to submit one GPU job with all eight cells per seed.'
    exit 0
fi
exec bash "$HERE/submit.sh" --ablate "$@"
