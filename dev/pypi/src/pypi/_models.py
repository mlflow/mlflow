from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version


def _parse_upload_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_requires_python(dists: list[dict[str, Any]]) -> SpecifierSet | None:
    raw = next((rp for d in dists if (rp := d.get("requires_python"))), None)
    if not raw:
        return None
    try:
        return SpecifierSet(raw)
    except InvalidSpecifier:
        return None


@dataclass(frozen=True)
class Release:
    version: Version
    upload_time: datetime | None
    yanked: bool
    requires_python: SpecifierSet | None


@dataclass(frozen=True)
class Package:
    name: str
    releases: tuple[Release, ...]

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Package:
        info = data.get("info") or {}
        raw_releases = data.get("releases") or {}
        releases: list[Release] = []
        for raw_version, dists in raw_releases.items():
            try:
                version = Version(raw_version)
            except InvalidVersion:
                # Skip versions PyPI accepts but `packaging` rejects
                # (e.g., https://pypi.org/project/pytz/2004d).
                continue
            times = [t for d in dists if (t := _parse_upload_time(d.get("upload_time_iso_8601")))]
            releases.append(
                Release(
                    version=version,
                    upload_time=min(times) if times else None,
                    yanked=len(dists) > 0 and all(d.get("yanked", False) for d in dists),
                    requires_python=_parse_requires_python(dists),
                )
            )

        releases.sort(key=lambda r: r.version)
        return cls(name=info.get("name", ""), releases=tuple(releases))

    @property
    def versions(self) -> tuple[Version, ...]:
        return tuple(r.version for r in self.releases)

    @property
    def latest_version(self) -> Version | None:
        """Highest non-pre, non-dev, non-yanked version, or None."""
        stable = [
            r.version
            for r in self.releases
            if not r.yanked and not r.version.is_prerelease and not r.version.is_devrelease
        ]
        return max(stable) if stable else None

    def get_release(self, version: str | Version) -> Release | None:
        target = version if isinstance(version, Version) else Version(version)
        return next((r for r in self.releases if r.version == target), None)
