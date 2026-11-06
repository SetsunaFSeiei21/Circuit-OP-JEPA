"""Unbuffered, nested logging with stderr kept on the stderr stream."""
import contextlib
import sys
from pathlib import Path


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, value):
        self.console.write(value)
        self.file.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def isatty(self):
        return False


@contextlib.contextmanager
def log_to(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', buffering=1) as file:
        with contextlib.redirect_stdout(Tee(sys.stdout, file)), contextlib.redirect_stderr(Tee(sys.stderr, file)):
            yield
