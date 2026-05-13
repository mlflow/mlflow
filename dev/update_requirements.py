"""
This script updates the `max_major_version` attribute of each package in a YAML dependencies
specification (e.g. requirements/core-requirements.yaml) to the maximum available version on PyPI.
"""

import asyncio
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import yaml
from pypi import Package, get_packages

PACKAGE_NAMES = ["tracing", "skinny", "core", "gateway"]
RELEASE_CUTOFF_DAYS = 14
PYPI_URL = os.environ.get("PYPI_URL", "https://pypi.org").rstrip("/")


def check_pypi_accessibility() -> None:
    try:
        with urllib.request.urlopen(PYPI_URL, timeout=5):
            pass
    except (urllib.error.URLError, OSError):
        raise SystemExit(
            f"Error: Cannot connect to {PYPI_URL}. "
            "If it's not accessible, set the PYPI_URL environment variable to a PyPI proxy URL."
        )


def get_latest_major_version(package: Package) -> int | None:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=RELEASE_CUTOFF_DAYS)
    versions = [
        r.version
        for r in package.releases
        if not r.yanked
        and not r.version.is_devrelease
        and not r.version.is_prerelease
        and r.upload_time < cutoff
    ]
    return max(versions).major if versions else None


def update_max_major_version(raw: str, key: str, old_value: int, new_value: int) -> str:
    """
    Update the max_major_version value for a specific package using regex.
    This preserves comments and formatting exactly as they appear in the file.
    """
    # Use word boundaries to ensure we match the exact number, not a substring
    pattern = rf"(^{re.escape(key)}:.*?max_major_version:)\s+\b{old_value}\b"
    updated, count = re.subn(
        pattern, rf"\1 {new_value}", raw, count=1, flags=re.DOTALL | re.MULTILINE
    )
    if count == 0:
        raise ValueError(
            f"Failed to update {key}.max_major_version from {old_value} to {new_value}. "
            "The pattern may not match the YAML structure."
        )
    return updated


def main() -> None:
    check_pypi_accessibility()

    file_to_requirements = {}
    file_to_src = {}
    pip_releases: set[str] = set()
    for package_name in PACKAGE_NAMES:
        req_file_path = os.path.join("requirements", package_name + "-requirements.yaml")
        with open(req_file_path) as f:
            requirements_src = f.read()
        requirements = yaml.safe_load(requirements_src)
        file_to_requirements[req_file_path] = requirements
        file_to_src[req_file_path] = requirements_src
        for req_info in requirements.values():
            if not req_info.get("freeze", False):
                pip_releases.add(req_info["pip_release"])

    sorted_releases = sorted(pip_releases)
    packages = dict(zip(sorted_releases, asyncio.run(get_packages(sorted_releases))))

    for req_file_path, requirements in file_to_requirements.items():
        updated_src = file_to_src[req_file_path]
        changes_made = False
        for key, req_info in requirements.items():
            pip_release = req_info["pip_release"]
            max_major_version = req_info["max_major_version"]
            if req_info.get("freeze", False):
                continue
            latest_major_version = get_latest_major_version(packages[pip_release])
            if latest_major_version is None:
                print(f"Skipping {key}: no releases older than {RELEASE_CUTOFF_DAYS}d found")
                continue
            if latest_major_version > max_major_version:
                updated_src = update_max_major_version(
                    updated_src, key, max_major_version, latest_major_version
                )
                print(f"Updated {key}.max_major_version to {latest_major_version}")
                changes_made = True

        if changes_made:
            with open(req_file_path, "w") as f:
                f.write(updated_src)


if __name__ == "__main__":
    main()
