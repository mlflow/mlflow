import functools
import typing as t

import requests
from packaging.specifiers import SpecifierSet
from packaging import version

DEV_VERSION = "dev"
# Treat "dev" as "newer than any existing versions"
DEV_NUMERIC = "9999.9999.9999"


class Version(version.Version):
    def __init__(self, version: str) -> None:
        # Override the constructor to handle "dev"
        super().__init__(DEV_NUMERIC if version == DEV_VERSION else version)


@functools.lru_cache()
def get_released_versions(package_name: str) -> t.List[Version]:
    url = "https://pypi.python.org/pypi/{}/json".format(package_name)
    resp = requests.get(url)
    resp.raise_for_status()
    return [
        Version(version)
        for version, dist_files in resp.json()["releases"].items()
        # Ignore:
        # - Releases without distribution files
        # - Yanked releases
        if len(dist_files) > 0 and (not dist_files[0].get("yanked", False))
    ]


def select_latest_micro_versions(versions: t.List[Version]) -> t.List[Version]:
    seen = set()
    micro_versions = []
    # Sort versions in descending order and select the first appearance of each minor version
    # --------
    # 3.2.0 <-
    # 3.1.0
    # 3.0.0
    # 2.5.1 <-
    # 2.5.0
    # 2.4.0 <-
    # ...
    for ver in sorted(versions, reverse=True):
        major_and_minor = ver.release[:2]
        if major_and_minor not in seen:
            seen.add(major_and_minor)
            micro_versions.append(ver)
    return sorted(micro_versions)


def filter_versions(
    versions: t.List[Version],
    min_ver: Version,
    max_ver: Version,
    exclude: t.List[Version],
) -> t.List[Version]:
    if exclude is None:
        exclude = []
    filtered_versions: t.List[Version] = []
    for v in versions:
        if (
            v not in exclude
            and not (v.is_devrelease or v.is_prerelease)
            and v <= max_ver
            and v >= min_ver
        ):
            filtered_versions.append(v)
    return filtered_versions


def get_extra_requirements(
    requirements: t.Optional[t.Dict[str, t.List[str]]],
    version: Version,
) -> t.List[str]:
    if requirements is None:
        return []

    if isinstance(requirements, dict):
        for specifiers, reqs in requirements.items():
            specifier_set = SpecifierSet(specifiers.replace(DEV_VERSION, DEV_NUMERIC))
            if specifier_set.contains(version):
                return reqs
        return []

    raise TypeError(f"Invalid type for `requirements`: {type(requirements)}")
