from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pypi import Package

from flavors._schema import Version

RELEASE_CUTOFF_DAYS = 7


def get_released_versions(package: Package) -> list[Version]:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=RELEASE_CUTOFF_DAYS)
    versions: list[Version] = []
    for release in package.releases:
        if release.yanked or release.upload_time >= cutoff:
            continue
        if release.version.is_devrelease or release.version.is_prerelease:
            continue
        versions.append(Version(str(release.version), release.upload_time))
    return versions
