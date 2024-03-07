import re
from pathlib import Path
from typing import List, Union

import click
from packaging.version import Version


def get_current_py_version() -> str:
    text = Path("mlflow", "version.py").read_text()
    return re.search(r'VERSION = "(.+)"', text).group(1)


def replace_dev_suffix_with(version, repl):
    return re.sub(r"\.dev0$", repl, version)


def replace_occurrences(files: List[Path], pattern: Union[str, re.Pattern], repl: str) -> None:
    if not isinstance(pattern, re.Pattern):
        pattern = re.compile(pattern)
    for f in files:
        old_text = f.read_text()
        if not pattern.search(old_text):
            continue
        new_text = pattern.sub(repl, old_text)
        f.write_text(new_text)


def update_versions(new_py_version: str) -> None:
    """
    `new_py_version` is either a release version (e.g. "2.1.0") or a dev version
    (e.g. "2.1.0.dev0").
    """
    current_py_version = get_current_py_version()
    current_py_version_without_suffix = replace_dev_suffix_with(current_py_version, "")

    # Python
    replace_occurrences(
        files=[Path("mlflow", "version.py")],
        pattern=re.escape(current_py_version),
        repl=new_py_version,
    )

    # pyproject.toml
    replace_occurrences(
        files=[Path("pyproject.toml"), Path("pyproject.skinny.toml")],
        pattern=re.compile(r'^version\s+=\s+".+"$', re.MULTILINE),
        repl=f'version = "{new_py_version}"',
    )

    # JS
    replace_occurrences(
        files=[
            Path(
                "mlflow",
                "server",
                "js",
                "src",
                "common",
                "constants.tsx",
            )
        ],
        pattern=re.escape(current_py_version),
        repl=new_py_version,
    )

    # Java
    for java_extension in ["xml", "java"]:
        replace_occurrences(
            files=Path("mlflow", "java").rglob(f"*.{java_extension}"),
            pattern=rf"{re.escape(current_py_version_without_suffix)}(-SNAPSHOT)?",
            repl=replace_dev_suffix_with(new_py_version, "-SNAPSHOT"),
        )

    # R
    replace_occurrences(
        files=[Path("mlflow", "R", "mlflow", "DESCRIPTION")],
        pattern=f"Version: {re.escape(current_py_version_without_suffix)}",
        repl=f"Version: {replace_dev_suffix_with(new_py_version, '')}",
    )


def validate_new_version(
    ctx: click.Context,
    param: click.Parameter,
    value: str,
) -> str:
    new = Version(value)
    current = Version(get_current_py_version())
    if new < current:
        raise click.BadParameter(
            f"New version {new} is not greater than or equal to current version {current}"
        )
    return value


@click.group()
def update_mlflow_versions():
    pass


@update_mlflow_versions.command(
    help="""
Update MLflow package versions BEFORE release.

Usage:

python dev/update_mlflow_versions.py pre-release --new-version 1.29.0
"""
)
@click.option(
    "--new-version", callback=validate_new_version, required=True, help="New version to release"
)
def pre_release(new_version: str):
    update_versions(new_py_version=new_version)


@update_mlflow_versions.command(
    help="""
Update MLflow package versions AFTER release.

Usage:

python dev/update_mlflow_versions.py post-release --new-version 1.29.0
"""
)
@click.option(
    "--new-version",
    callback=validate_new_version,
    required=True,
    help="New version that was released",
)
def post_release(new_version: str):
    current_version = Version(get_current_py_version())
    msg = (
        "It appears you ran this command on a release branch because the current version "
        f"({current_version}) is not a dev version. Please re-run this command on the master "
        "branch."
    )
    assert current_version.is_devrelease, msg
    new_version = Version(new_version)
    # Increment the patch version and append ".dev0"
    new_py_version = f"{new_version.major}.{new_version.minor}.{new_version.micro + 1}.dev0"
    update_versions(new_py_version=new_py_version)


if __name__ == "__main__":
    update_mlflow_versions()
