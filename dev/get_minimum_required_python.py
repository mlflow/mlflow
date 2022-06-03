"""
A script to automatically find the minimum required python version for a specified package.

Usage:
python dev/get_minimum_required_python.py -p scikit-learn -v 1.1.0 --python-versions "3.7,3.8"
"""
import requests
from packaging.version import Version
from packaging.specifiers import SpecifierSet
import typing as t
import argparse


def get_requires_python(package: str, version: str) -> t.Optional[str]:
    resp = requests.get(f"https://pypi.python.org/pypi/{package}/json")
    resp.raise_for_status()
    return next(
        (
            distributions[0].get("requires_python")
            for ver, distributions in resp.json()["releases"].items()
            if ver == version and distributions
        ),
        None,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--package", help="Package name", required=True)
    parser.add_argument("-v", "--version", help="Package version", required=True)
    parser.add_argument(
        "--python-versions",
        help=(
            "Comma separated string representing python versions. "
            "If `requires_python` is unavailable for a specified package, "
            "the minimum version will be selected."
        ),
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sorted_python_versions = sorted(args.python_versions.split(","), key=Version)
    min_python_version = sorted_python_versions[0]
    requires_python = get_requires_python(args.package, args.version)
    if not requires_python:
        print(min_python_version)
        return

    specifier_set = SpecifierSet(requires_python)
    matched_versions = list(filter(specifier_set.contains, sorted_python_versions))
    print(matched_versions[0] if matched_versions else min_python_version)


if __name__ == "__main__":
    main()
