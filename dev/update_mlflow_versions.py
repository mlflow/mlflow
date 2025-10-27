import argparse
import logging
import re
from pathlib import Path

from packaging.version import Version

_logger = logging.getLogger(__name__)

_PYTHON_VERSION_FILES = [
    Path("mlflow", "version.py"),
]

_PYPROJECT_TOML_FILES = [
    Path("pyproject.toml"),
    Path("pyproject.release.toml"),
    Path("libs/skinny/pyproject.toml"),
    Path("libs/tracing/pyproject.toml"),
]

_JAVA_VERSION_FILES = Path("mlflow", "java").rglob("*.java")

_JAVA_POM_XML_FILES = Path("mlflow", "java").rglob("*.xml")

_JS_VERSION_FILES = [
    Path(
        "mlflow",
        "server",
        "js",
        "src",
        "common",
        "constants.tsx",
    )
]

_R_VERSION_FILES = [Path("mlflow", "R", "mlflow", "DESCRIPTION")]


def get_current_py_version() -> str:
    text = Path("mlflow", "version.py").read_text()
    return re.search(r'VERSION = "(.+)"', text).group(1)


def get_java_py_version_pattern(version: str) -> str:
    version_without_suffix = replace_dev_or_rc_suffix_with(version, "")
    return rf"{re.escape(version_without_suffix)}(-SNAPSHOT)?"


def get_java_new_py_version(new_py_version: str) -> str:
    return replace_dev_or_rc_suffix_with(new_py_version, "-SNAPSHOT")


def replace_dev_or_rc_suffix_with(version, repl):
    parsed = Version(version)
    base_version = parsed.base_version
    return base_version + repl if parsed.is_prerelease else version


def replace_occurrences(files: list[Path], pattern: str | re.Pattern, repl: str) -> None:
    if not isinstance(pattern, re.Pattern):
        pattern = re.compile(pattern)
    for f in files:
        old_text = f.read_text()
        if not pattern.search(old_text):
            continue
        new_text = pattern.sub(repl, old_text)
        f.write_text(new_text)


def replace_python(old_version: str, new_py_version: str, paths: list[Path]) -> None:
    replace_occurrences(
        files=paths,
        pattern=re.escape(old_version),
        repl=new_py_version,
    )


def replace_pyproject_toml(new_py_version: str, paths: list[Path]) -> None:
    replace_occurrences(
        files=paths,
        pattern=re.compile(r'^version\s+=\s+".+"$', re.MULTILINE),
        repl=f'version = "{new_py_version}"',
    )
    # Update mlflow-skinny and mlflow-tracing versions to match the new mlflow version.
    replace_occurrences(
        files=paths,
        pattern=re.compile(r"^\s*\"mlflow-skinny==.+\",$", re.MULTILINE),
        repl=f'  "mlflow-skinny=={new_py_version}",',
    )
    replace_occurrences(
        files=paths,
        pattern=re.compile(r"^\s*\"mlflow-tracing==.+\",$", re.MULTILINE),
        repl=f'  "mlflow-tracing=={new_py_version}",',
    )


def replace_js(old_version: str, new_py_version: str, paths: list[Path]) -> None:
    replace_occurrences(
        files=paths,
        pattern=re.escape(old_version),
        repl=new_py_version,
    )


def replace_java(old_version: str, new_py_version: str, paths: list[Path]) -> None:
    old_py_version_pattern = get_java_py_version_pattern(old_version)
    dev_suffix_replaced = get_java_new_py_version(new_py_version)

    replace_occurrences(
        files=paths,
        pattern=old_py_version_pattern,
        repl=dev_suffix_replaced,
    )


# Note: the pom.xml files define versions of dependencies as
# well. this causes issues when the mlflow version matches the
# version of a dependency. to work around, we make sure to
# match only the correct keys
def replace_java_pom_xml(old_version: str, new_py_version: str, paths: list[Path]) -> None:
    old_py_version_pattern = get_java_py_version_pattern(old_version)
    dev_suffix_replaced = get_java_new_py_version(new_py_version)

    mlflow_version_tag_pattern = r"<mlflow.version>"
    mlflow_spark_pattern = r"<artifactId>mlflow-spark_2\.1[23]</artifactId>\s+<version>"
    mlflow_parent_pattern = r"<artifactId>mlflow-parent</artifactId>\s+<version>"

    # combine the three tags together to form the regex
    mlflow_replace_pattern = (
        rf"({mlflow_version_tag_pattern}|{mlflow_spark_pattern}|{mlflow_parent_pattern})"
        + f"{old_py_version_pattern}"
        + r"(</mlflow.version>|</version>)"
    )

    # group 1: everything before the version
    # group 2: optional -SNAPSHOT
    # group 3: everything after the version
    replace_str = f"\\g<1>{dev_suffix_replaced}\\g<3>"

    replace_occurrences(
        files=paths,
        pattern=mlflow_replace_pattern,
        repl=replace_str,
    )


def replace_r(old_py_version: str, new_py_version: str, paths: list[Path]) -> None:
    current_py_version_without_suffix = replace_dev_or_rc_suffix_with(old_py_version, "")

    replace_occurrences(
        files=paths,
        pattern=f"Version: {re.escape(current_py_version_without_suffix)}",
        repl=f"Version: {replace_dev_or_rc_suffix_with(new_py_version, '')}",
    )


def update_versions(new_py_version: str) -> None:
    """
    `new_py_version` is either:
      - a release version (e.g. "2.1.0")
      - a RC version (e.g. "2.1.0rc0")
      - a dev version (e.g. "2.1.0.dev0")
    """
    old_py_version = get_current_py_version()

    replace_python(old_py_version, new_py_version, _PYTHON_VERSION_FILES)
    replace_pyproject_toml(new_py_version, _PYPROJECT_TOML_FILES)
    replace_js(old_py_version, new_py_version, _JS_VERSION_FILES)
    replace_java(old_py_version, new_py_version, _JAVA_VERSION_FILES)
    replace_java_pom_xml(old_py_version, new_py_version, _JAVA_POM_XML_FILES)
    replace_r(old_py_version, new_py_version, _R_VERSION_FILES)


def validate_new_version(value: str) -> str:
    new = Version(value)
    current = Version(get_current_py_version())

    # this could be the case if we just promoted an RC to a release
    if new < current:
        _logger.warning(
            f"New version {new} is not greater than or equal to current version {current}. "
            "If the previous version was an RC, this is expected. If not, please make sure the "
            "specified new version is correct."
        )
        # exit with 0 to avoid failing the CI job
        exit(0)

    return value


def pre_release(new_version: str):
    """
    Update MLflow package versions BEFORE release.

    Usage:

    python dev/update_mlflow_versions.py pre-release --new-version 1.29.0
    """
    validate_new_version(new_version)
    update_versions(new_py_version=new_version)


def post_release(new_version: str):
    """
    Update MLflow package versions AFTER release.

    Usage:

    python dev/update_mlflow_versions.py post-release --new-version 1.29.0
    """
    validate_new_version(new_version)
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
    parser = argparse.ArgumentParser(description="Update MLflow package versions")
    subparsers = parser.add_subparsers(dest="command", help="Command to run", required=True)

    # pre-release subcommand
    pre_parser = subparsers.add_parser(
        "pre-release",
        help="Update MLflow package versions BEFORE release",
    )
    pre_parser.add_argument("--new-version", required=True, help="New version to release")

    # post-release subcommand
    post_parser = subparsers.add_parser(
        "post-release",
        help="Update MLflow package versions AFTER release",
    )
    post_parser.add_argument("--new-version", required=True, help="New version that was released")

    args = parser.parse_args()

    if args.command == "pre-release":
        pre_release(args.new_version)
    elif args.command == "post-release":
        post_release(args.new_version)
