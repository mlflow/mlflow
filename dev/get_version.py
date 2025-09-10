import subprocess
from pathlib import Path


def read_version(path: Path) -> str:
    with path.open() as f:
        for line in f:
            if line.startswith("version ="):
                return line.split("=")[-1].strip().strip('"').strip("'")
    raise ValueError(f"Version not found in {path}")


def main():
    repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    pyproject_path = Path(repo_root) / "pyproject.toml"
    version = read_version(pyproject_path)
    print(version)


if __name__ == "__main__":
    main()
