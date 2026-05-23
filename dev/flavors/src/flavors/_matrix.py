"""
Generate the cross-version test matrix from `mlflow/ml-package-versions.yml`.

# Usage

```
# Test all items
flavors matrix

# Exclude items for dev versions
flavors matrix --no-dev

# Test items affected by config file updates
flavors matrix --ref-versions-yaml /path/to/ref-versions.yml

# Test items affected by flavor module updates
flavors matrix --changed-files "mlflow/sklearn/__init__.py"

# Test a specific flavor
flavors matrix --flavors sklearn

# Test a specific version
flavors matrix --versions 1.1.1
```
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, TypeVar

import requests
from packaging.specifiers import SpecifierSet
from pydantic import BaseModel, ConfigDict
from pypi import Package, get_packages

from flavors._loader import VERSIONS_YAML_PATH, load, load_or_default
from flavors._releases import get_released_versions
from flavors._schema import DEV_NUMERIC, DEV_VERSION, FlavorConfig, PackageInfo, Version

T = TypeVar("T")


class MatrixItem(BaseModel):
    name: str
    flavor: str
    category: str
    job_name: str
    install: str
    run: str
    package: str
    version: Version
    python: str
    java: str
    supported: bool
    free_disk_space: bool
    runs_on: str
    pre_test: str | None = None
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    def __hash__(self) -> int:
        return hash(frozenset(dict(self)))


def get_latest_micro_versions(versions: list[Version]) -> list[Version]:
    """
    Returns the latest micro version in each minor version.
    """
    by_minor: dict[tuple[int, ...], Version] = {}
    for ver in sorted(versions, reverse=True):
        by_minor.setdefault(ver.release[:2], ver)
    return list(by_minor.values())


def filter_versions(
    flavor: str,
    versions: list[Version],
    min_ver: Version,
    max_ver: Version,
    unsupported: list[SpecifierSet],
    allow_unreleased_max_version: bool = False,
) -> list[Version]:
    """
    Returns the versions that satisfy the following conditions:
    1. Newer than or equal to `min_ver`.
    2. Older than or equal to `max_ver.major`.
    3. Not in `unsupported`.
    """

    def _is_supported(v: Version) -> bool:
        for specified_set in unsupported:
            if v in specified_set:
                return False
        return True

    def _check_max(v: Version) -> bool:
        return v <= max_ver or (
            # Exclude versions uploaded very recently to avoid testing unstable or potentially
            # buggy releases. Newly released versions may have unresolved issues
            # (see: https://github.com/huggingface/transformers/issues/34370).
            v.major <= max_ver.major
            and v.days_since_release is not None
            and v.days_since_release >= 1
        )

    def _check_min(v: Version) -> bool:
        return v >= min_ver

    return [v for v in versions if _check_min(v) and _check_max(v) and _is_supported(v)]


FLAVOR_FILE_PATTERN = re.compile(r"^(mlflow|tests)/(.+?)(_autolog(ging)?)?(\.py|/)")


def get_changed_flavors(changed_files: list[str], flavors: set[str]) -> set[str]:
    """
    Detects changed flavors from a list of changed files.
    """
    changed_flavors: set[str] = set()
    for f in changed_files:
        match = FLAVOR_FILE_PATTERN.match(f)
        if match and match.group(2) in flavors:
            changed_flavors.add(match.group(2))
    return changed_flavors


def _find_matches(spec: dict[str, T], version: str) -> Iterator[T]:
    """
    Args:
        spec: A dictionary with key as version specifier and value as the corresponding value.
            For example, {"< 1.0.0": "numpy<2.0", ">= 1.0.0": "numpy>=2.0"}.
        version: The version to match against the specifiers.

    Returns:
        An iterator of values that match the version.
    """
    for specifier, val in spec.items():
        specifier_set = SpecifierSet(specifier.replace(DEV_VERSION, DEV_NUMERIC))
        if specifier_set.contains(DEV_NUMERIC if version == DEV_VERSION else version):
            yield val


def get_matched_requirements(requirements: dict[str, list[str]], version: str) -> list[str]:
    if not isinstance(requirements, dict):
        raise TypeError(
            f"Invalid object type for `requirements`: '{type(requirements)}'. Must be dict."
        )
    reqs: set[str] = set()
    for packages in _find_matches(requirements, version):
        reqs.update(packages)
    return sorted(reqs)


def get_java_version(java: dict[str, str] | None, version: str) -> str:
    return _get_spec_value(java, version, "17")


def _requires_python_from_repo(repo_url: str) -> str | None:
    """
    Fetch requires-python from repository's pyproject.toml for dev version inference.
    """
    match = re.match(r"https://github\.com/([^/]+/[^/]+)/tree/HEAD(?:/(.+))?", repo_url)
    if not match:
        raise ValueError(f"Invalid GitHub repository URL format: {repo_url}")

    owner_repo = match.group(1)
    subpath = match.group(2) or ""
    pyproject_path = f"{subpath}/pyproject.toml" if subpath else "pyproject.toml"
    raw_url = f"https://raw.githubusercontent.com/{owner_repo}/HEAD/{pyproject_path}"

    print(f"Fetching pyproject.toml from {owner_repo} (path: {pyproject_path})", file=sys.stderr)

    try:
        resp = requests.get(raw_url, timeout=10)
        resp.raise_for_status()
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            print(f"  pyproject.toml not found at {raw_url}", file=sys.stderr)
            return None
        raise

    if match := re.search(r'requires-python\s*=\s*["\']([^"\']+)["\']', resp.text):
        print(f"  Found requires-python: {match.group(1)}", file=sys.stderr)
        return match.group(1)

    print("  requires-python field not found in pyproject.toml", file=sys.stderr)
    return None


def infer_python_version(package: Package, version: str, repo_url: str | None = None) -> str:
    """
    Infer the minimum Python version required by the package.
    """
    candidates = ("3.10", "3.11")

    if version == DEV_VERSION:
        # `Version("dev")` would raise InvalidVersion, so resolve dev separately
        # via the repo's pyproject.toml when available.
        if repo_url and (rp := _requires_python_from_repo(repo_url)):
            spec = SpecifierSet(rp)
            return next(filter(spec.contains, candidates), candidates[0])
        return candidates[0]

    if (release := package.get_release(version)) and release.requires_python:
        return next(filter(release.requires_python.contains, candidates), candidates[0])

    return candidates[0]


def _get_spec_value(spec: dict[str, str] | None, version: str, default: str) -> str:
    if spec and (match := next(_find_matches(spec, version), None)):
        return match
    return default


def get_python_version(
    python: dict[str, str] | None, package: Package, version: str, repo_url: str | None = None
) -> str:
    if python and (match := next(_find_matches(python, version), None)):
        return match

    return infer_python_version(package, version, repo_url)


def get_runs_on(runs_on: dict[str, str] | None, version: str) -> str:
    return _get_spec_value(runs_on, version, "ubuntu-latest")


def remove_comments(s: str) -> str:
    return "\n".join(l for l in s.strip().split("\n") if not l.strip().startswith("#"))


def make_pip_install_command(packages: list[str]) -> str:
    return "uv pip install --system " + " ".join(f"'{x}'" for x in packages)


def divider(title: str, length: int | None = None) -> str:
    length = length or shutil.get_terminal_size(fallback=(80, 24))[0]
    return "\n" + f" {title} ".center(length, "=") + "\n"


def split_by_comma(x: str) -> list[str]:
    return [s for item in x.split(",") if (s := item.strip())]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--versions-yaml",
        required=False,
        default=VERSIONS_YAML_PATH,
        help=(
            f"Local file path of the config yaml. Defaults to '{VERSIONS_YAML_PATH}' "
            "on the branch where this script is running."
        ),
    )
    parser.add_argument(
        "--ref-versions-yaml",
        required=False,
        default=None,
        help=(
            "Local file path of the reference config yaml which will be compared with the "
            "config specified by `--versions-yaml` in order to identify the config updates."
        ),
    )
    parser.add_argument(
        "--changed-files",
        type=lambda x: [] if x.strip() == "" else x.strip().split("\n"),
        required=False,
        default=None,
        help=("A string that represents a list of changed files"),
    )

    parser.add_argument(
        "--flavors",
        required=False,
        type=split_by_comma,
        help=(
            "Comma-separated string specifying which flavors to test (e.g. 'sklearn, xgboost'). "
            "If unspecified, all flavors are tested."
        ),
    )
    parser.add_argument(
        "--versions",
        required=False,
        type=split_by_comma,
        help=(
            "Comma-separated string specifying which versions to test (e.g. '1.2.3, 4.5.6'). "
            "If unspecified, all versions are tested."
        ),
    )
    parser.add_argument(
        "--no-dev",
        action="store_true",
        default=False,
        help="If True, exclude dev versions in the test matrix.",
    )
    parser.add_argument(
        "--only-latest",
        action="store_true",
        default=False,
        help=(
            "If True, only test the latest version of each group. Useful when you want to avoid "
            "running too many GitHub Action jobs."
        ),
    )


def parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set a test matrix for the cross version tests")
    add_arguments(parser)
    return parser.parse_args(args)


FLAVOR_NAME_ALIASES = {"pytorch-lightning": "pytorch"}


def validate_test_coverage(flavor: str, config: FlavorConfig) -> None:
    """
    Validate that all test files for the flavor are executed in the cross-version tests.

    This is done by parsing `run` commands in the `ml-package-versions.yml` to get the list
    of executed test files, and then comparing it with the actual test files in the directory.
    """
    test_dir = os.path.join("tests", flavor)
    tested_files = set()

    for category, cfg in config.categories:
        if not cfg.run:
            continue

        # Consolidate multi-line commands with "\" to a single line
        commands = cfg.run.replace("\\\n", "").split("\n")

        # Parse pytest commands to get the executed test files
        for cmd in commands:
            cmd = cmd.strip().rstrip(";")
            if cmd.startswith("pytest"):
                tested_files |= _get_test_files_from_pytest_command(cmd, test_dir)

    if untested_files := _get_test_files(test_dir) - tested_files:
        # TODO: Update this after fixing ml-package-versions.yml to
        # have all test files in the matrix.
        warnings.warn(
            f"Flavor '{flavor}' has test files that are not covered by the test matrix. \n"
            + "\n".join(f"\033[91m - {t}\033[0m" for t in untested_files)
            + f"\nPlease update {VERSIONS_YAML_PATH} to execute all test files. Note that this "
            "check does not handle complex syntax in test commands e.g. loop. It is generally "
            "recommended to use simple commands as we cannot test the test commands themselves."
        )


PYTEST_FILE_PATTERN = re.compile(r"^test_.*\.py$")


def _get_test_files(test_dir_or_path: str) -> set[Path]:
    """List all test files in the given directory or file path."""
    path = Path(test_dir_or_path)
    if path.is_dir():
        return set(path.rglob("test_*.py"))

    if PYTEST_FILE_PATTERN.match(path.name):
        return {path}

    return set()


def _get_test_files_from_pytest_command(cmd: str, test_dir: str) -> set[Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ignore", action="append")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_known_args(shlex.split(cmd))[0]

    executed_files: set[Path] = set()
    ignore_files: set[Path] = set()
    for path in args.paths:
        if path.startswith(test_dir):
            executed_files |= _get_test_files(path)
    for ignore_path in args.ignore or []:
        if ignore_path.startswith(test_dir):
            ignore_files |= _get_test_files(ignore_path)
    return executed_files - ignore_files


def validate_requirements(
    requirements: dict[str, list[str]],
    name: str,
    category: str,
    package_info: PackageInfo,
    versions: list[Version],
) -> None:
    """
    Validate that the requirements specified in the config don't contain unused items.
    Here's an example of invalid requirements:

    ```
    sklearn:
        package_info:
            pip_release: "scikit-learn"
        autologging:
            minimum: "1.3.0"
            maximum: "1.5.0"
            requirements:
                "< 1.0.0": ["numpy<2.0"]    # Unused
                ">= 1.4.0": ["numpy>=2.0"]  # Used
    ```
    """
    for specifier in requirements:
        if "dev" in specifier and package_info.install_dev:
            continue

        # Does this version specifier (e.g. '< 1.0.0') match at least one version?
        # If not, raise an error.
        spec_set = SpecifierSet(specifier)
        if not any(map(spec_set.contains, versions)):
            raise ValueError(
                f"Found unused requirements {specifier!r} for {name} / {category}. "
                "Please remove it or adjust the version specifier."
            )


async def expand_config(
    config: dict[str, FlavorConfig], *, is_ref: bool = False
) -> set[MatrixItem]:
    matrix: set[MatrixItem] = set()
    pip_releases = list({fc.package_info.pip_release for fc in config.values()})
    packages = dict(zip(pip_releases, await get_packages(pip_releases)))
    for name, flavor_config in config.items():
        flavor = FLAVOR_NAME_ALIASES.get(name, name)
        package_info = flavor_config.package_info
        package = packages[package_info.pip_release]
        all_versions = get_released_versions(package)
        free_disk_space = package_info.pip_release in (
            "transformers",
            "sentence-transformers",
            "torch",
        )
        validate_test_coverage(name, flavor_config)
        for category, cfg in flavor_config.categories:
            versions = filter_versions(
                flavor,
                all_versions,
                cfg.minimum,
                cfg.maximum,
                cfg.unsupported or [],
                allow_unreleased_max_version=cfg.allow_unreleased_max_version or False,
            )
            versions = get_latest_micro_versions(versions)

            # Test every n minor versions if specified
            if cfg.test_every_n_versions > 1:
                versions = sorted(versions)[:: -cfg.test_every_n_versions][::-1]

            # Always test the minimum version
            if cfg.minimum not in versions and cfg.minimum in all_versions:
                versions.append(cfg.minimum)

            if not is_ref and cfg.requirements:
                validate_requirements(cfg.requirements, name, category, package_info, versions)

            for ver in versions:
                requirements = [f"{package_info.pip_release}=={ver}"]
                requirements.extend(get_matched_requirements(cfg.requirements or {}, str(ver)))
                install = make_pip_install_command(requirements)
                run = remove_comments(cfg.run)
                python = get_python_version(cfg.python, package, str(ver), package_info.repo)
                runs_on = get_runs_on(cfg.runs_on, str(ver))
                java = get_java_version(cfg.java, str(ver))

                matrix.add(
                    MatrixItem(
                        name=name,
                        flavor=flavor,
                        category=category,
                        job_name=f"{name} / {category} / {ver}",
                        install=install,
                        run=run,
                        package=package_info.pip_release,
                        version=ver,
                        python=python,
                        java=java,
                        supported=ver <= cfg.maximum,
                        free_disk_space=free_disk_space,
                        runs_on=runs_on,
                        pre_test=cfg.pre_test,
                    )
                )

            # Add tracing SDK test with the latest stable version
            if len(versions) > 0 and category == "autologging" and cfg.test_tracing_sdk:
                version = max(versions)  # Test against the latest stable version
                matrix.add(
                    MatrixItem(
                        name=f"{name}-tracing",
                        flavor=flavor,
                        category="tracing-sdk",
                        job_name=f"{name} / tracing-sdk / {version}",
                        install=install,
                        # --import-mode=importlib is required for testing tracing SDK
                        # (mlflow-tracing) works properly, without being affected by environment.
                        run=run.replace("pytest", "pytest --import-mode=importlib"),
                        package=package_info.pip_release,
                        version=version,
                        java=java,
                        supported=version <= cfg.maximum,
                        free_disk_space=free_disk_space,
                        python=python,
                        runs_on=runs_on,
                    )
                )

            # Skip dev version testing: install_dev installs from git, which
            # doesn't respect UV_EXCLUDE_NEWER.
            if False:  # package_info.install_dev:
                install_dev = remove_comments(package_info.install_dev)
                if requirements := get_matched_requirements(cfg.requirements or {}, DEV_VERSION):
                    install = make_pip_install_command(requirements) + "\n" + install_dev
                else:
                    install = install_dev
                python = get_python_version(cfg.python, package, DEV_VERSION, package_info.repo)
                runs_on = get_runs_on(cfg.runs_on, DEV_VERSION)
                java = get_java_version(cfg.java, DEV_VERSION)

                run = remove_comments(cfg.run)
                dev_version = Version.create_dev()
                matrix.add(
                    MatrixItem(
                        name=name,
                        flavor=flavor,
                        category=category,
                        job_name=f"{name} / {category} / {dev_version}",
                        install=install,
                        run=run,
                        package=package_info.pip_release,
                        version=dev_version,
                        python=python,
                        java=java,
                        supported=False,
                        free_disk_space=free_disk_space,
                        runs_on=runs_on,
                        pre_test=cfg.pre_test,
                    )
                )
    return matrix


def apply_changed_files(changed_files: list[str], matrix: set[MatrixItem]) -> set[MatrixItem]:
    all_flavors = {x.flavor for x in matrix}
    changed_flavors = (
        # If matrix-generation code itself changed, re-run all tests.
        all_flavors
        if any(f.startswith("dev/flavors/") for f in changed_files)
        else get_changed_flavors(changed_files, all_flavors)
    )

    # Run langchain tests if any tracing files have been changed
    if any(f.startswith("mlflow/tracing/") for f in changed_files):
        changed_flavors.add("langchain")

    return set(filter(lambda x: x.flavor in changed_flavors, matrix))


async def _generate(args: argparse.Namespace) -> set[MatrixItem]:
    config = load(args.versions_yaml)
    if (args.ref_versions_yaml, args.changed_files).count(None) == 2:
        matrix = await expand_config(config)
    else:
        matrix = set()
        mat = await expand_config(config)

        if args.ref_versions_yaml:
            ref_config = load_or_default(args.ref_versions_yaml, default={})
            ref_matrix = await expand_config(ref_config, is_ref=True)
            matrix.update(mat.difference(ref_matrix))

        if args.changed_files:
            matrix.update(apply_changed_files(args.changed_files, mat))

    # Apply the filtering arguments
    if args.no_dev:
        dev_ver = Version.create_dev()
        matrix = {x for x in matrix if x.version != dev_ver}

    if args.flavors:
        matrix = {x for x in matrix if x.flavor in args.flavors}

    if args.versions:
        target_versions = list(map(Version, args.versions))
        matrix = {x for x in matrix if x.version in target_versions}

    if args.only_latest:
        groups: dict[tuple[str, str], list[MatrixItem]] = defaultdict(list)
        for item in matrix:
            groups[(item.name, item.category)].append(item)
        matrix = {max(group, key=lambda x: x.version) for group in groups.values()}

    return matrix


async def generate_matrix(args: list[str]) -> set[MatrixItem]:
    return await _generate(parse_args(args))


class CustomEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, MatrixItem):
            return o.model_dump(exclude_none=True)
        elif isinstance(o, Version):
            return str(o)
        return super().default(o)


def set_action_output(name: str, value: str) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"{name}={value}\n")


def split(matrix: list[MatrixItem], n: int) -> Iterator[list[MatrixItem]]:
    grouped_by_name: dict[str, list[MatrixItem]] = defaultdict(list)
    for item in matrix:
        grouped_by_name[item.name].append(item)

    num = len(matrix) // n
    chunk: list[MatrixItem] = []
    for group in grouped_by_name.values():
        chunk.extend(group)
        if len(chunk) >= num:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


async def run(args: argparse.Namespace) -> None:
    # https://docs.github.com/en/actions/learn-github-actions/usage-limits-billing-and-administration#usage-limits
    # > A job matrix can generate a maximum of 256 jobs per workflow run.
    MAX_ITEMS = 256
    NUM_JOBS = 2

    print(divider("Parameters"))
    print(json.dumps(vars(args), indent=2, default=str))
    matrix = sorted(await _generate(args), key=lambda x: (x.name, x.category, x.version))
    assert len(matrix) <= MAX_ITEMS * 2, f"Too many jobs: {len(matrix)} > {MAX_ITEMS * NUM_JOBS}"
    for idx, mat in enumerate(split(matrix, NUM_JOBS), start=1):
        payload = {"include": mat, "job_name": [x.job_name for x in mat]}
        print(divider(f"Matrix {idx}"))
        print(json.dumps(payload, indent=2, cls=CustomEncoder))
        if "GITHUB_ACTIONS" in os.environ:
            set_action_output(f"matrix{idx}", json.dumps(payload, cls=CustomEncoder))
            set_action_output(f"is_matrix{idx}_empty", "true" if len(mat) == 0 else "false")
