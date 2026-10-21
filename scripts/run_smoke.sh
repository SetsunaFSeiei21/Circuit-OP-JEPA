#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m unittest discover -s tests -v
python -m device_jepa smoke --output "${1:-runs/smoke}"
