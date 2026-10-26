"""Default locations for independent circuit experiments, including worktrees."""
from datetime import datetime
from itertools import count
import os
from pathlib import Path
import re
import subprocess


def results_root(project_root):
    """Share the primary checkout's runs/gpu_full directory across worktrees."""
    override = os.environ.get('JEPA_RESULTS_ROOT')
    if override:
        root = Path(override)
        if not root.is_absolute():
            raise ValueError('JEPA_RESULTS_ROOT must be an absolute directory')
        return root.resolve()

    project_root = Path(project_root).resolve()
    if not (project_root / '.git').exists():
        return project_root / 'runs' / 'gpu_full'
    # Git lists the primary checkout first, even when called in a linked worktree.
    # NUL separation also supports checkout paths containing spaces/newlines.
    try:
        listing = subprocess.check_output(
            ['git', '-C', str(project_root), 'worktree', 'list', '--porcelain', '-z'],
            stderr=subprocess.DEVNULL)
        first = os.fsdecode(listing).split('\0\0', 1)[0].split('\0')
        if first[0].startswith('worktree ') and 'bare' not in first:
            return Path(first[0][len('worktree '):]).resolve() / 'runs' / 'gpu_full'
    except (OSError, subprocess.CalledProcessError):
        pass
    if (project_root / '.git').is_file():
        raise ValueError('Cannot locate the primary checkout; set an absolute JEPA_RESULTS_ROOT')
    # Exported source trees without Git metadata use their own project directory.
    return project_root / 'runs' / 'gpu_full'


def create_default_output(project_root, circuit):
    """Reserve a new circuit/date directory atomically, never reuse an old run."""
    root = results_root(project_root)
    label = re.sub(r'[^\w.-]+', '_', circuit).strip('._') or 'circuit'
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    stem = f'{label}_{stamp}'
    root.mkdir(parents=True, exist_ok=True)
    for index in count(1):
        output = root / (stem if index == 1 else f'{stem}_{index}')
        try:
            output.mkdir()
        except FileExistsError:
            continue
        return output
