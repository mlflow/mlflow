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
import sys
import argparse
import json
import os
import re
import shutil
import functools
import typing as t
from collections import defaultdict

import yaml
import requests
from packaging.specifiers import SpecifierSet
from packaging.version import Version as OriginalVersion, InvalidVersion
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
    install_dev: t.Optional[str]


class TestConfig(BaseModel):
    minimum: Version
    maximum: Version
    unsupported: t.Optional[t.List[Version]]
    requirements: t.Optional[t.Dict[str, t.List[str]]]
    run: str
    allow_unreleased_max_version: t.Optional[bool]

    class Config:
        arbitrary_types_allowed = True

    @validator("minimum", pre=True)
    def validate_minimum(cls, v):  # pylint: disable=no-self-argument
        return Version(v)

    @validator("maximum", pre=True)
    def validate_maximum(cls, v):  # pylint: disable=no-self-argument
        return Version(v)

    @validator("unsupported", pre=True)
    def validate_unsupported(cls, v):  # pylint: disable=no-self-argument
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
    supported: bool

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
            print("Failed to read '{}' due to: `{}`".format(location, e))
            return if_error
        raise


@functools.lru_cache()
def get_released_versions(package_name):
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
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
            "Invalid object type for `requirements`: '{}'. Must be dict.".format(type(requirements))
        )

    reqs = set()
    for specifier, packages in requirements.items():
        specifier_set = SpecifierSet(specifier.replace(DEV_VERSION, DEV_NUMERIC))
        if specifier_set.contains(DEV_NUMERIC if version == DEV_VERSION else version):
            reqs = reqs.union(packages)
    return sorted(reqs)


def remove_comments(s):
    return "\n".join(l for l in s.strip().split("\n") if not l.strip().startswith("#"))


def make_pip_install_command(packages):
    return "pip install " + " ".join("'{}'".format(x) for x in packages)


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
                requirements = ["{}=={}".format(package_info.pip_release, ver)]
                requirements.extend(get_matched_requirements(cfg.requirements or {}, str(ver)))
                install = make_pip_install_command(requirements)
                run = remove_comments(cfg.run)

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
                        supported=ver <= cfg.maximum,
                    )
                )

            if package_info.install_dev:
                install_dev = remove_comments(package_info.install_dev)
                if cfg.requirements:
                    requirements = get_matched_requirements(cfg.requirements or {}, DEV_VERSION)
                    install = make_pip_install_command(requirements) + "\n" + install_dev
                else:
                    install = install_dev

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
                        supported=False,
                    )
                )
    return matrix


def apply_changed_files(changed_files, matrix):
    all_flavors = set(x.flavor for x in matrix)
    changed_flavors = (
        # If this file has been changed, re-run all tests
        all_flavors
        if (__file__ in changed_files)
        else get_changed_flavors(changed_files, all_flavors)
    )
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
            return dict(o)
        elif isinstance(o, Version):
            return str(o)
        return super().default(o)


def set_action_output(name, value):
    with open(os.getenv("GITHUB_OUTPUT"), "a") as f:
        f.write(f"{name}={value}\n")


def main(args):
    print(divider("Parameters"))
    print(json.dumps(args, indent=2))
    matrix = generate_matrix(args)
    is_matrix_empty = len(matrix) == 0
    matrix = sorted(matrix, key=lambda x: x.job_name)
    matrix = {"include": matrix, "job_name": [x.job_name for x in matrix]}

    print(divider("Matrix"))
    print(json.dumps(matrix, indent=2, cls=CustomEncoder))

    if "GITHUB_ACTIONS" in os.environ:
        set_action_output("matrix", json.dumps(matrix, cls=CustomEncoder))
        set_action_output("is_matrix_empty", "true" if is_matrix_empty else "false")


if __name__ == "__main__":
    main(sys.argv[1:])
