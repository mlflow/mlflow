import functools
import subprocess
from pathlib import Path


@functools.cache
def get_repo_root() -> Path:
    out = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        timeout=5,
    )
    return Path(out.strip())
