# /// script
# dependencies = [
#   "tomli",
# ]
# ///
import subprocess
from pathlib import Path

import tomli


def main():
    repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    pyproject_path = Path(repo_root) / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomli.load(f)

    print(pyproject["project"]["version"])


if __name__ == "__main__":
    main()
