#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
CAMPAIGN="${CAMPAIGN:-}" # result root containing campaign_pin.json
[[ -n "$CAMPAIGN" ]] || { echo 'Set CAMPAIGN to the multi-circuit result root.' >&2; exit 2; }
cd "$PROJECT_ROOT"
exec "${JEPA_PYTHON:-python}" -m final_exam.multi_circuit.worker_pin summary --campaign "$CAMPAIGN"
