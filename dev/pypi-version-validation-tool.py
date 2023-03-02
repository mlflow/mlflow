"""
This script performs manual validation checks on any packages that are released on pypi against
those defined in our requirements files. It will print out invalid configurations if any are
encountered.
"""

from ruamel.yaml import YAML
import requests
from packaging.version import Version, InvalidVersion


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

    reqs_files = ["requirements/core-requirements.yaml", "requirements/skinny-requirements.yaml"]
    validation = True
    for reqs in reqs_files:

        with open(reqs) as f:
            requirements_src = f.read()
            requirements = yaml.load(requirements_src)

        for key, req_info in requirements.items():
            pip_release = req_info["pip_release"]
            max_major_version = req_info["max_major_version"]
            latest_major_version = get_latest_major_version(pip_release)

            if latest_major_version < max_major_version:
                validation = False
                print(
                    f"version mismatch for requirements file '{reqs}' for package '{pip_release}'."
                    f"\n\tLatest_major version supported on pypi: {latest_major_version}."
                    f"\n\tMax major version defined in requirements file: {max_major_version}."
                )
    if validation:
        print("All package versions are within pypi boundaries!")


if __name__ == "__main__":
    main()
