"""
This script updates the `max_major_version` attribute of each package in a YAML dependencies
specification (e.g. requirements/core-requirements.yaml) to the maximum available version on PyPI.
"""

import os

import requests
from packaging.version import InvalidVersion, Version
from ruamel.yaml import YAML

PACKAGE_NAMES = ["tracing", "skinny", "core", "gateway"]


def get_latest_major_version(package_name: str) -> int:
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

    return max(versions).major


def main():
    yaml = YAML()
    yaml.preserve_quotes = True

    for package_name in PACKAGE_NAMES:
        req_file_path = os.path.join("requirements", package_name + "-requirements.yaml")
        with open(req_file_path) as f:
            requirements_src = f.read()
            requirements = yaml.load(requirements_src)

        changes_made = False
        for key, req_info in requirements.items():
            pip_release = req_info["pip_release"]
            max_major_version = req_info["max_major_version"]
            if req_info.get("freeze", False):
                continue
            latest_major_version = get_latest_major_version(pip_release)
            if latest_major_version > max_major_version:
                requirements[key]["max_major_version"] = latest_major_version
                print(f"Updated {key}.max_major_version to {latest_major_version}")
                changes_made = True

        if changes_made:
            with open(req_file_path, "w") as f:
                yaml.dump(requirements, f)


if __name__ == "__main__":
    main()
