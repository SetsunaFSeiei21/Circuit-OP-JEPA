"""Best-effort source metadata; training does not require a Git installation."""
import subprocess
import sys


def code_revision(root):
    """Return the checkout revision, or an explicit sentinel if Git is unavailable."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True,
            text=True, encoding='utf-8', errors='replace', check=False, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        reason = type(error).__name__
    else:
        revision = result.stdout.strip()
        if result.returncode == 0 and revision:
            return revision
        reason = f'git exited with status {result.returncode}' if result.returncode else 'empty Git output'
    print(f'Warning: code_commit=unavailable ({reason}); continuing without Git revision metadata.',
          file=sys.stderr, flush=True)
    return 'unavailable'
