# /// script
# dependencies = [
#   "toml",
# ]
# ///
import subprocess
from pathlib import Path

import toml


def main():
    repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    pyproject_path = Path(repo_root) / "pyproject.toml"
    with open(pyproject_path) as f:
        pyproject = toml.load(f)

    print(pyproject["project"]["version"])


if __name__ == "__main__":
    main()
