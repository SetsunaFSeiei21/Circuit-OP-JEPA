#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${JEPA_PYTHON:-python}" "$PROJECT_ROOT/scripts/run/run_comparison.py" "$@"
