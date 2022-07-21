import argparse
import difflib
import re

import yaml
import requests
from packaging.version import Version, InvalidVersion


def get_latest_major_version(package_name: str, minium_version: str = None) -> int:
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    min_ver = Version(minium_version if minium_version else "0.0.0")
    versions = []
    for version, distributions in data["releases"].items():
        if len(distributions) == 0 or any(d.get("yanked", False) for d in distributions):
            continue

        try:
            version = Version(version)
        except InvalidVersion:
            # Ignore invalid versions such as https://pypi.org/project/pytz/2004d
            continue

        if version.is_devrelease or version.is_prerelease or version < min_ver:
            continue

        versions.append(version)

    return max(versions).major + 1  # +1 for testing, will be removed


def replace_max_major_version(yaml_string: str, pip_release: str, max_major_version: int) -> str:
    pattern = r"""
  pip_release: {pip_release}
  (.*?)max_major_version: \d+
""".format(
        pip_release=pip_release
    )
    repl = r"""
  pip_release: {pip_release}
  \1max_major_version: {max_major_version}
""".format(
        pip_release=pip_release, max_major_version=max_major_version
    )
    assert re.search(pattern, yaml_string, flags=re.DOTALL)
    return re.sub(pattern, repl, yaml_string, flags=re.DOTALL)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update max major versions in the requirements.yaml specification"
    )
    parser.add_argument(
        "--requirements-yaml-location",
        required=True,
        help="Local file path of the requirements.yaml specification.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="If specified, only check if the requirements.yaml specification is up to date.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.requirements_yaml_location, "r") as f:
        requirements_src = f.read()
        requirements = yaml.safe_load(requirements_src)

    new_requirements_src = requirements_src
    for req_info in requirements.values():
        pip_release = req_info["pip_release"]
        max_major_version = req_info["max_major_version"]
        minium_version = req_info.get("minium_version")
        latest_major_version = get_latest_major_version(pip_release, minium_version)
        assert latest_major_version >= max_major_version
        if latest_major_version != max_major_version:
            new_requirements_src = replace_max_major_version(
                new_requirements_src, pip_release, latest_major_version
            )
            print(
                f"Updated {pip_release} max_major_version"
                f" {max_major_version} -> {latest_major_version}"
            )

    if new_requirements_src != requirements_src:
        diff = difflib.ndiff(
            requirements_src.splitlines(keepends=True),
            new_requirements_src.splitlines(keepends=True),
        )
        print("========== Diff ==========")
        print("".join(diff), end="")
        print("==========================")
        if args.check:
            print(f"{args.requirements_yaml_location} is not up to date")
            exit(1)

        with open(args.requirements_yaml_location, "w") as f:
            f.write(new_requirements_src)
    else:
        print(f"{args.requirements_yaml_location} is up to date!")


if __name__ == "__main__":
    main()
