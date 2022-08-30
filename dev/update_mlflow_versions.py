import re
from pathlib import Path
from typing import List

import click
from packaging.version import Version


def get_current_version() -> str:
    text = Path("mlflow", "version.py").read_text()
    ver = Version(re.search(r'VERSION = "(.+)"', text).group(1))
    return f"{ver.major}.{ver.minor}.{ver.micro}"


def replace_occurrences(files: List[Path], pattern: str, repl: str) -> None:
    pattern = re.compile(pattern)
    for f in files:
        old_text = f.read_text()
        assert pattern.search(old_text), f"Pattern {pattern} not found in {f}"
        new_text = pattern.sub(repl, old_text)
        f.write_text(new_text)


def update_versions(new_version: str, is_dev_version: bool) -> None:
    current_version = re.escape(get_current_version())
    # Java
    suffix = "-SNAPSHOT" if is_dev_version else ""
    replace_occurrences(
        files=Path("mlflow", "java").rglob("*.xml"),
        pattern=rf"{current_version}(-SNAPSHOT)?",
        repl=new_version + suffix,
    )
    # Python
    suffix = ".dev0" if is_dev_version else ""
    replace_occurrences(
        files=[Path("mlflow", "version.py")],
        pattern=rf"{current_version}(\.dev0)?",
        repl=new_version + suffix,
    )
    # JS
    suffix = ".dev0" if is_dev_version else ""
    replace_occurrences(
        files=[
            Path(
                "mlflow",
                "server",
                "js",
                "src",
                "common",
                "constants.js",
            )
        ],
        pattern=rf"{current_version}(\.dev0)?",
        repl=new_version + suffix,
    )
    # R
    replace_occurrences(
        files=[Path("mlflow", "R", "mlflow", "DESCRIPTION")],
        pattern=current_version,
        repl=new_version,
    )


@click.group()
def update_mlflow_versions():
    pass


@update_mlflow_versions.command(help="Update MLflow package versions BEFORE release")
@click.option("--new-version", required=True, help="New version to release")
def before_release(new_version: str):
    update_versions(new_version, is_dev_version=False)


@update_mlflow_versions.command(help="Update MLflow package versions AFTER release")
@click.option("--new-version", required=True, help="New version that was released")
def after_release(new_version: str):
    new_version = Version(new_version)
    next_new_version = f"{new_version.major}.{new_version.minor}.{new_version.micro + 1}"
    update_versions(next_new_version, is_dev_version=True)


if __name__ == "__main__":
    update_mlflow_versions()
