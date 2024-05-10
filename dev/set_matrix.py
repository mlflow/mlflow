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
import shutil
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests
import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion
from packaging.version import Version as OriginalVersion
from pydantic import BaseModel, validator

VERSIONS_YAML_PATH = "mlflow/ml-package-versions.yml"
DEV_VERSION = "dev"
# Treat "dev" as "newer than any existing versions"
DEV_NUMERIC = "9999.9999.9999"


class Version(OriginalVersion):
    def __init__(self, version):
        self._is_dev = version == DEV_VERSION
        super().__init__(DEV_NUMERIC if self._is_dev else version)

    def __str__(self):
        return DEV_VERSION if self._is_dev else super().__str__()

    @classmethod
    def create_dev(cls):
        return cls(DEV_VERSION)


class PackageInfo(BaseModel):
    pip_release: str
    install_dev: Optional[str] = None


class TestConfig(BaseModel):
    minimum: Version
    maximum: Version
    unsupported: Optional[List[Version]] = None
    requirements: Optional[Dict[str, List[str]]] = None
    python: Optional[Dict[str, str]] = None
    java: Optional[Dict[str, str]] = None
    run: str
    allow_unreleased_max_version: Optional[bool] = None
    pre_test: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    @validator("minimum", pre=True)
    def validate_minimum(cls, v):
        return Version(v)

    @validator("maximum", pre=True)
    def validate_maximum(cls, v):
        return Version(v)

    @validator("unsupported", pre=True)
    def validate_unsupported(cls, v):
        return [Version(v) for v in v] if v else None


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
    pre_test: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __hash__(self):
        return hash(frozenset(dict(self)))


def read_yaml(location, if_error=None):
    try:
        if re.match(r"^https?://", location):
            resp = requests.get(location)
            resp.raise_for_status()
            return yaml.safe_load(resp.text)
        else:
            with open(location) as f:
                return yaml.safe_load(f)
    except Exception as e:
        if if_error is not None:
            print(f"Failed to read '{location}' due to: `{e}`")
            return if_error
        raise


def get_released_versions(package_name):
    data = pypi_json(package_name)
    versions = []
    for version, distributions in data["releases"].items():
        if len(distributions) == 0 or any(d.get("yanked", False) for d in distributions):
            continue

        try:
            version = Version(version)
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
    versions, min_ver, max_ver, unsupported=None, allow_unreleased_max_version=False
):
    """
    Returns the versions that satisfy the following conditions:
    1. Newer than or equal to `min_ver`.
    2. Older than or equal to `max_ver.major`.
    3. Not in `unsupported`.
    """
    unsupported = unsupported or []
    # Prevent specifying non-existent versions
    assert min_ver in versions
    assert max_ver in versions or allow_unreleased_max_version
    assert all(v in versions for v in unsupported)

    def _is_not_unsupported(v):
        return v not in unsupported

    def _is_older_than_or_equal_to_max_major_version(v):
        return v.major <= max_ver.major

    def _is_newer_than_or_equal_to_min_version(v):
        return v >= min_ver

    return list(
        functools.reduce(
            lambda vers, f: filter(f, vers),
            [
                _is_not_unsupported,
                _is_older_than_or_equal_to_max_major_version,
                _is_newer_than_or_equal_to_min_version,
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


def get_java_version(java: Optional[Dict[str, str]], version: str) -> str:
    default = "11"
    if java is None:
        return default

    for specifier, java_ver in java.items():
        specifier_set = SpecifierSet(specifier.replace(DEV_VERSION, DEV_NUMERIC))
        if specifier_set.contains(DEV_NUMERIC if version == DEV_VERSION else version):
            return java_ver

    return default


@functools.lru_cache(maxsize=128)
def pypi_json(package: str) -> Dict[str, Any]:
    resp = requests.get(f"https://pypi.org/pypi/{package}/json")
    resp.raise_for_status()
    return resp.json()


def get_requires_python(package: str, version: str) -> str:
    package_json = pypi_json(package)
    requires_python = next(
        (
            distributions[0].get("requires_python")
            for ver, distributions in package_json["releases"].items()
            if ver == version and distributions
        ),
        None,
    )
    candidates = ("3.8", "3.9")
    if requires_python is None:
        return candidates[0]

    spec = SpecifierSet(requires_python)
    return next((c for c in candidates if spec.contains(c)), None) or candidates[0]


def get_python_version(python: Optional[Dict[str, str]], package: str, version: str) -> str:
    if python:
        for specifier, py_ver in python.items():
            specifier_set = SpecifierSet(specifier.replace(DEV_VERSION, DEV_NUMERIC))
            if specifier_set.contains(DEV_NUMERIC if version == DEV_VERSION else version):
                return py_ver

    return get_requires_python(package, version)


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


def expand_config(config):
    matrix = set()
    for name, cfgs in config.items():
        flavor = get_flavor(name)
        package_info = PackageInfo(**cfgs.pop("package_info"))
        all_versions = get_released_versions(package_info.pip_release)
        free_disk_space = package_info.pip_release in (
            "transformers",
            "sentence-transformers",
            "torch",
        )
        for category, cfg in cfgs.items():
            cfg = TestConfig(**cfg)
            versions = filter_versions(
                all_versions,
                cfg.minimum,
                cfg.maximum,
                cfg.unsupported or [],
                allow_unreleased_max_version=cfg.allow_unreleased_max_version or False,
            )
            versions = get_latest_micro_versions(versions)

            # Always test the minimum version
            if cfg.minimum not in versions:
                versions.append(cfg.minimum)

            for ver in versions:
                requirements = [f"{package_info.pip_release}=={ver}"]
                requirements.extend(get_matched_requirements(cfg.requirements or {}, str(ver)))
                install = make_pip_install_command(requirements)
                run = remove_comments(cfg.run)
                python = get_python_version(cfg.python, package_info.pip_release, str(ver))
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
                        pre_test=cfg.pre_test,
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
                        pre_test=cfg.pre_test,
                    )
                )
    return matrix


def apply_changed_files(changed_files, matrix):
    all_flavors = {x.flavor for x in matrix}
    changed_flavors = (
        # If this file has been changed, re-run all tests
        all_flavors
        if (__file__ in changed_files)
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
            ref_matrix = expand_config(ref_config)
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


def validate_action_config(num_jobs: int):
    with open(".github/workflows/cross-version-tests.yml") as f:
        s = f.read()
    s = re.sub(
        r"needs\.set-matrix\.outputs\.matrix\d",
        "needs.set-matrix.outputs.matrix",
        s,
    )
    s = re.sub(
        r"needs\.set-matrix\.outputs\.is_matrix\d_empty",
        "needs.set-matrix.outputs.is_matrix_empty",
        s,
    )
    jobs = yaml.safe_load(s)["jobs"]
    jobs = [v for name, v in jobs.items() if name.startswith("test")]
    assert len(jobs) == num_jobs, f"Expected {num_jobs} jobs, but got {len(jobs)}"
    assert all(jobs[0] == j for j in jobs[1:]), "All jobs must have the same configuration"


def main(args):
    # https://docs.github.com/en/actions/learn-github-actions/usage-limits-billing-and-administration#usage-limits
    # > A job matrix can generate a maximum of 256 jobs per workflow run.
    MAX_ITEMS = 256
    NUM_JOBS = 2
    validate_action_config(NUM_JOBS)

    print(divider("Parameters"))
    print(json.dumps(args, indent=2))
    matrix = generate_matrix(args)
    matrix = sorted(matrix, key=lambda x: (x.name, x.category, x.version))
    matrix = [x for x in matrix if x.flavor not in ("gluon", "mleap")]
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
