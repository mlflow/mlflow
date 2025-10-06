"""
A script to set a matrix for the cross version tests for MLflow Models / autologging integrations.

# Usage:

```
# Test all items
python dev/set_matrix.py

# Exclude items for dev versions
python dev/set_matrix.py --no-dev

# Test items affected by config file updates
python dev/set_matrix.py --ref-versions-yaml \
    "https://raw.githubusercontent.com/mlflow/mlflow/master/ml-package-versions.yml"

# Test items affected by flavor module updates
python dev/set_matrix.py --changed-files "mlflow/sklearn/__init__.py"

# Test a specific flavor
python dev/set_matrix.py --flavors sklearn

# Test a specific version
python dev/set_matrix.py --versions 1.1.1
```
"""

import argparse
import functools
import json
import os
import re
import shlex
import shutil
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TypeVar

import requests
import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion
from packaging.version import Version as OriginalVersion
from pydantic import BaseModel, ConfigDict, field_validator

VERSIONS_YAML_PATH = "mlflow/ml-package-versions.yml"
DEV_VERSION = "dev"
# Treat "dev" as "newer than any existing versions"
DEV_NUMERIC = "9999.9999.9999"

T = TypeVar("T")


class Version(OriginalVersion):
    def __init__(self, version: str, release_date: datetime | None = None):
        self._is_dev = version == DEV_VERSION
        self._release_date = release_date
        super().__init__(DEV_NUMERIC if self._is_dev else version)

    def __str__(self):
        return DEV_VERSION if self._is_dev else super().__str__()

    @classmethod
    def create_dev(cls):
        return cls(DEV_VERSION, datetime.now(timezone.utc))

    @property
    def days_since_release(self) -> int | None:
        """
        Compute the number of days since this version was released.
        Returns None if release date is not available.
        """
        if self._release_date is None:
            return None
        delta = datetime.now(timezone.utc) - self._release_date
        return delta.days


class PackageInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pip_release: str
    install_dev: str | None = None
    module_name: str | None = None
    genai: bool = False


class TestConfig(BaseModel):
    minimum: Version
    maximum: Version
    unsupported: list[SpecifierSet] | None = None
    requirements: dict[str, list[str]] | None = None
    python: dict[str, str] | None = None
    runs_on: dict[str, str] | None = None
    java: dict[str, str] | None = None
    run: str
    allow_unreleased_max_version: bool | None = None
    pre_test: str | None = None
    test_every_n_versions: int = 1
    test_tracing_sdk: bool = False
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_validator("minimum", mode="before")
    @classmethod
    def validate_minimum(cls, v):
        return Version(v)

    @field_validator("maximum", mode="before")
    @classmethod
    def validate_maximum(cls, v):
        return Version(v)

    @field_validator("unsupported", mode="before")
    @classmethod
    def validate_unsupported(cls, v):
        return [SpecifierSet(x) for x in v] if v else None

    @field_validator("python", mode="before")
    @classmethod
    def validate_python_requirements(cls, v):
        if v is None:
            return v

        # Read the minimum Python version from .python-version file
        python_version_file = Path(".python-version")
        min_python_version = python_version_file.read_text().strip()

        # Check if any value in the python dict matches the minimum version
        for version in v.values():
            if version == min_python_version:
                raise ValueError(f"Unnecessary Python version requirement: {version}")

        return v


class FlavorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    package_info: PackageInfo
    models: TestConfig | None = None
    autologging: TestConfig | None = None

    @property
    def categories(self) -> list[tuple[str, TestConfig]]:
        cs = []
        if self.models:
            cs.append(("models", self.models))
        if self.autologging:
            cs.append(("autologging", self.autologging))
        return cs


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

    def __hash__(self):
        return hash(frozenset(dict(self)))


def read_yaml(location, if_error=None):
    try:
        if re.match(r"^https?://", location):
            resp = requests.get(location)
            resp.raise_for_status()
            yaml_dict = yaml.safe_load(resp.text)
        else:
            with open(location) as f:
                yaml_dict = yaml.safe_load(f)
        return {name: FlavorConfig(**cfg) for name, cfg in yaml_dict.items()}
    except Exception as e:
        if if_error is not None:
            print(f"Failed to read '{location}' due to: `{e}`")
            return if_error
        raise


def uploaded_recently(dist: dict[str, Any]) -> bool:
    if ut := dist.get("upload_time_iso_8601"):
        delta = datetime.now(timezone.utc) - datetime.fromisoformat(ut.replace("Z", "+00:00"))
        return delta.days < 1
    return False


def get_released_versions(package_name: str) -> list[Version]:
    data = pypi_json(package_name)
    versions: list[Version] = []
    for version_str, distributions in data["releases"].items():
        if len(distributions) == 0 or any(d.get("yanked", False) for d in distributions):
            continue

        # Extract the earliest upload time as the release date
        upload_times = []
        for dist in distributions:
            if ut := dist.get("upload_time_iso_8601"):
                upload_times.append(datetime.fromisoformat(ut.replace("Z", "+00:00")))

        release_date = min(upload_times) if upload_times else None
        try:
            version = Version(version_str, release_date)
        except InvalidVersion:
            # Ignore invalid versions such as https://pypi.org/project/pytz/2004d
            continue

        if version.is_devrelease or version.is_prerelease:
            continue

        versions.append(version)

    return versions


def get_latest_micro_versions(versions):
    """
    Returns the latest micro version in each minor version.
    """
    seen = set()
    latest_micro_versions = []
    for ver in sorted(versions, reverse=True):
        major_and_minor = ver.release[:2]
        if major_and_minor not in seen:
            seen.add(major_and_minor)
            latest_micro_versions.append(ver)
    return latest_micro_versions


def filter_versions(
    flavor: str,
    versions: list[Version],
    min_ver: Version,
    max_ver: Version,
    unsupported: list[SpecifierSet],
    allow_unreleased_max_version: bool = False,
):
    """
    Returns the versions that satisfy the following conditions:
    1. Newer than or equal to `min_ver`.
    2. Older than or equal to `max_ver.major`.
    3. Not in `unsupported`.
    """

    def _is_supported(v):
        for specified_set in unsupported:
            if v in specified_set:
                return False
        return True

    def _check_max(v: Version) -> bool:
        return v <= max_ver or (
            # Exclude versions uploaded very recently to avoid testing unstable or potentially
            # buggy releases. Newly released versions may have unresolved issues
            # (see: https://github.com/huggingface/transformers/issues/34370).
            v.major <= max_ver.major and v.days_since_release and v.days_since_release >= 1
        )

    def _check_min(v: Version) -> bool:
        return v >= min_ver

    return list(
        functools.reduce(
            lambda vers, f: filter(f, vers),
            [
                _is_supported,
                _check_max,
                _check_min,
            ],
            versions,
        )
    )


FLAVOR_FILE_PATTERN = re.compile(r"^(mlflow|tests)/(.+?)(_autolog(ging)?)?(\.py|/)")


def get_changed_flavors(changed_files, flavors):
    """
    Detects changed flavors from a list of changed files.
    """
    changed_flavors = set()
    for f in changed_files:
        match = FLAVOR_FILE_PATTERN.match(f)
        if match and match.group(2) in flavors:
            changed_flavors.add(match.group(2))
    return changed_flavors


def get_matched_requirements(requirements, version=None):
    if not isinstance(requirements, dict):
        raise TypeError(
            f"Invalid object type for `requirements`: '{type(requirements)}'. Must be dict."
        )

    reqs = set()
    for specifier, packages in requirements.items():
        specifier_set = SpecifierSet(specifier.replace(DEV_VERSION, DEV_NUMERIC))
        if specifier_set.contains(DEV_NUMERIC if version == DEV_VERSION else version):
            reqs = reqs.union(packages)
    return sorted(reqs)


def get_java_version(java: dict[str, str] | None, version: str) -> str:
    if java and (match := next(_find_matches(java, version), None)):
        return match

    return "17"


@functools.lru_cache(maxsize=128)
def pypi_json(package: str) -> dict[str, Any]:
    resp = requests.get(f"https://pypi.org/pypi/{package}/json")
    resp.raise_for_status()
    return resp.json()


def _requires_python(package: str, version: str) -> str | None:
    package_json = pypi_json(package)
    for ver, dist in package_json.get("releases", {}).items():
        if ver != version:
            continue

        for d in dist:
            if rp := d.get("requires_python"):
                return rp
    return None


def infer_python_version(package: str, version: str) -> str:
    """
    Infer the minimum Python version required by the package.
    """
    candidates = ("3.10", "3.11")
    if rp := _requires_python(package, version):
        spec = SpecifierSet(rp)
        return next(filter(spec.contains, candidates), candidates[0])

    return candidates[0]


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


def get_python_version(python: dict[str, str] | None, package: str, version: str) -> str:
    if python and (match := next(_find_matches(python, version), None)):
        return match

    return infer_python_version(package, version)


def get_runs_on(runs_on: dict[str, str] | None, version: str) -> str:
    if runs_on and (match := next(_find_matches(runs_on, version), None)):
        return match

    return "ubuntu-latest"


def remove_comments(s):
    return "\n".join(l for l in s.strip().split("\n") if not l.strip().startswith("#"))


def make_pip_install_command(packages):
    return "pip install " + " ".join(f"'{x}'" for x in packages)


def divider(title, length=None):
    length = shutil.get_terminal_size(fallback=(80, 24))[0] if length is None else length
    rest = length - len(title) - 2
    left = rest // 2 if rest % 2 else (rest + 1) // 2
    return "\n{} {} {}\n".format("=" * left, title, "=" * (rest - left))


def split_by_comma(x):
    stripped = x.strip()
    return list(map(str.strip, stripped.split(","))) if stripped != "" else []


def parse_args(args):
    parser = argparse.ArgumentParser(description="Set a test matrix for the cross version tests")
    parser.add_argument(
        "--versions-yaml",
        required=False,
        default="mlflow/ml-package-versions.yml",
        help=(
            "URL or local file path of the config yaml. Defaults to "
            "'mlflow/ml-package-versions.yml' on the branch where this script is running."
        ),
    )
    parser.add_argument(
        "--ref-versions-yaml",
        required=False,
        default=None,
        help=(
            "URL or local file path of the reference config yaml which will be compared with the "
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

    return parser.parse_args(args)


def get_flavor(name):
    return {"pytorch-lightning": "pytorch"}.get(name, name)


def validate_test_coverage(flavor: str, config: FlavorConfig):
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
        commands = []
        curr = ""
        for cmd in cfg.run.split("\n"):
            if cmd.endswith("\\"):
                curr += cmd.rstrip("\\")
            else:
                commands.append(curr + cmd)
                curr = ""

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


def _get_test_files_from_pytest_command(cmd, test_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ignore", action="append")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_known_args(shlex.split(cmd))[0]

    executed_files = set()
    ignore_files = set()
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


def expand_config(config: dict[str, Any], *, is_ref: bool = False) -> set[MatrixItem]:
    matrix = set()
    for name, flavor_config in config.items():
        flavor = get_flavor(name)
        package_info = flavor_config.package_info
        all_versions = get_released_versions(package_info.pip_release)
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
                python = get_python_version(cfg.python, package_info.pip_release, str(ver))
                runs_on = get_runs_on(cfg.runs_on, ver)
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
                version = sorted(versions)[-1]  # Test against the latest stable version
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

            if package_info.install_dev:
                install_dev = remove_comments(package_info.install_dev)
                requirements = get_matched_requirements(cfg.requirements or {}, DEV_VERSION)
                if requirements:
                    install = make_pip_install_command(requirements) + "\n" + install_dev
                else:
                    install = install_dev
                python = get_python_version(cfg.python, package_info.pip_release, DEV_VERSION)
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


def apply_changed_files(changed_files, matrix):
    all_flavors = {x.flavor for x in matrix}
    changed_flavors = (
        # If this file has been changed, re-run all tests
        all_flavors
        if str(Path(__file__).relative_to(Path.cwd())) in changed_files
        else get_changed_flavors(changed_files, all_flavors)
    )

    # Run langchain tests if any tracing files have been changed
    if any(f.startswith("mlflow/tracing/") for f in changed_files):
        changed_flavors.add("langchain")

    return set(filter(lambda x: x.flavor in changed_flavors, matrix))


def generate_matrix(args):
    args = parse_args(args)
    config = read_yaml(args.versions_yaml)
    if (args.ref_versions_yaml, args.changed_files).count(None) == 2:
        matrix = expand_config(config)
    else:
        matrix = set()
        mat = expand_config(config)

        if args.ref_versions_yaml:
            ref_config = read_yaml(args.ref_versions_yaml, if_error={})
            ref_matrix = expand_config(ref_config, is_ref=True)
            matrix.update(mat.difference(ref_matrix))

        if args.changed_files:
            matrix.update(apply_changed_files(args.changed_files, mat))

    # Apply the filtering arguments
    if args.no_dev:
        matrix = filter(lambda x: x.version != Version.create_dev(), matrix)

    if args.flavors:
        matrix = filter(lambda x: x.flavor in args.flavors, matrix)

    if args.versions:
        matrix = filter(lambda x: x.version in map(Version, args.versions), matrix)

    if args.only_latest:
        groups = defaultdict(list)
        for item in matrix:
            groups[(item.name, item.category)].append(item)
        matrix = {max(group, key=lambda x: x.version) for group in groups.values()}

    return set(matrix)


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, MatrixItem):
            return o.model_dump(exclude_none=True)
        elif isinstance(o, Version):
            return str(o)
        return super().default(o)


def set_action_output(name, value):
    with open(os.getenv("GITHUB_OUTPUT"), "a") as f:
        f.write(f"{name}={value}\n")


def split(matrix, n):
    grouped_by_name = defaultdict(list)
    for item in matrix:
        grouped_by_name[item.name].append(item)

    num = len(matrix) // n
    chunk = []
    for group in grouped_by_name.values():
        chunk.extend(group)
        if len(chunk) >= num:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


def main(args):
    # https://docs.github.com/en/actions/learn-github-actions/usage-limits-billing-and-administration#usage-limits
    # > A job matrix can generate a maximum of 256 jobs per workflow run.
    MAX_ITEMS = 256
    NUM_JOBS = 2

    print(divider("Parameters"))
    print(json.dumps(args, indent=2))
    matrix = generate_matrix(args)
    matrix = sorted(matrix, key=lambda x: (x.name, x.category, x.version))
    assert len(matrix) <= MAX_ITEMS * 2, f"Too many jobs: {len(matrix)} > {MAX_ITEMS * NUM_JOBS}"
    for idx, mat in enumerate(split(matrix, NUM_JOBS), start=1):
        mat = {"include": mat, "job_name": [x.job_name for x in mat]}
        print(divider(f"Matrix {idx}"))
        print(json.dumps(mat, indent=2, cls=CustomEncoder))
        if "GITHUB_ACTIONS" in os.environ:
            set_action_output(f"matrix{idx}", json.dumps(mat, cls=CustomEncoder))
            set_action_output(f"is_matrix{idx}_empty", "true" if len(mat) == 0 else "false")


if __name__ == "__main__":
    main(sys.argv[1:])
