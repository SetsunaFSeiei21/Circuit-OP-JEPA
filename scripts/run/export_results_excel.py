#!/usr/bin/env python3
"""Export an existing experiment to Excel without importing training packages."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from device_jepa.excel import main


if __name__ == '__main__':
    main()
