#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/smoke
if (( $# > 0 )) && [[ "$1" != --* ]]; then OUT="$1"; shift; fi
exec "${JEPA_PYTHON:-python}" -m device_jepa smoke --output "$OUT" "$@"
