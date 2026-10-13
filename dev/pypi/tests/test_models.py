from __future__ import annotations

from typing import Any

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pypi._models import Package

_DEFAULT_UPLOAD = "2024-01-01T00:00:00Z"


def _payload(releases: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    # Inject a default upload_time into every non-empty dist so tests don't
    # have to spell it out — `from_json` skips releases that lack one.
    normalized = {
        v: [{"upload_time_iso_8601": _DEFAULT_UPLOAD, **d} for d in dists]
        for v, dists in releases.items()
    }
    return {"info": {"name": "demo", "summary": "demo package"}, "releases": normalized}


def test_versions_are_parsed_to_packaging_version() -> None:
    pkg = Package.from_json(
        _payload({
            "1.0.0": [{"upload_time_iso_8601": "2024-01-01T00:00:00Z"}],
            "2.0.0": [{"upload_time_iso_8601": "2024-06-01T00:00:00Z"}],
        })
    )
    assert all(isinstance(v, Version) for v in pkg.versions)
    assert pkg.versions == (Version("1.0.0"), Version("2.0.0"))


def test_releases_without_distributions_are_dropped() -> None:
    pkg = Package.from_json(
        _payload({
            "1.0.0": [{}],
            "0.5.0": [],  # all distributions deleted on PyPI
        })
    )
    assert pkg.versions == (Version("1.0.0"),)


def test_releases_without_upload_time_are_dropped() -> None:
    raw = {
        "info": {"name": "demo"},
        "releases": {
            "1.0.0": [{"upload_time_iso_8601": "2024-01-01T00:00:00Z"}],
            "0.5.0": [{"filename": "demo-0.5.0.tar.gz"}],  # missing upload_time
        },
    }
    pkg = Package.from_json(raw)
    assert pkg.versions == (Version("1.0.0"),)


def test_invalid_versions_are_dropped() -> None:
    pkg = Package.from_json(
        _payload({
            "1.0.0": [{"upload_time_iso_8601": "2024-01-01T00:00:00Z"}],
            "2004d": [{"upload_time_iso_8601": "2024-01-01T00:00:00Z"}],
        })
    )
    assert pkg.versions == (Version("1.0.0"),)


@pytest.mark.parametrize(
    ("releases", "expected"),
    [
        ({"1.0.0": [{}], "2.0.0": [{}]}, Version("2.0.0")),
        ({"1.0.0": [{}], "2.0.0a1": [{}]}, Version("1.0.0")),
        ({"1.0.0": [{}], "2.0.0.dev1": [{}]}, Version("1.0.0")),
        ({"1.0.0": [{"yanked": True}], "0.9.0": [{}]}, Version("0.9.0")),
        ({}, None),
    ],
)
def test_latest_version_skips_pre_dev_and_yanked(
    releases: dict[str, list[dict[str, Any]]], expected: Version | None
) -> None:
    pkg = Package.from_json(_payload(releases))
    assert pkg.latest_version == expected


def test_get_release_accepts_str_or_version() -> None:
    pkg = Package.from_json(_payload({"1.2.3": [{}]}))
    by_str = pkg.get_release("1.2.3")
    by_version = pkg.get_release(Version("1.2.3"))
    assert by_str is not None
    assert by_str is by_version
    assert pkg.get_release("9.9.9") is None


def test_release_upload_time_picks_earliest() -> None:
    pkg = Package.from_json(
        _payload({
            "1.0.0": [
                {"upload_time_iso_8601": "2024-03-01T00:00:00Z"},
                {"upload_time_iso_8601": "2024-01-01T00:00:00Z"},
            ],
        })
    )
    release = pkg.get_release("1.0.0")
    assert release is not None
    assert release.upload_time is not None
    assert release.upload_time.year == 2024
    assert release.upload_time.month == 1


def test_release_yanked_only_when_all_distributions_yanked() -> None:
    pkg = Package.from_json(_payload({"1.0.0": [{"yanked": True}, {"yanked": False}]}))
    release = pkg.get_release("1.0.0")
    assert release is not None
    assert release.yanked is False


def test_release_yanked_when_every_distribution_yanked() -> None:
    pkg = Package.from_json(_payload({"1.0.0": [{"yanked": True}, {"yanked": True}]}))
    release = pkg.get_release("1.0.0")
    assert release is not None
    assert release.yanked is True


def test_release_requires_python_parsed_to_specifier_set() -> None:
    pkg = Package.from_json(_payload({"1.0.0": [{"requires_python": ">=3.10,<4"}]}))
    release = pkg.get_release("1.0.0")
    assert release is not None
    assert release.requires_python == SpecifierSet(">=3.10,<4")
    assert Version("3.11") in release.requires_python
    assert Version("3.9") not in release.requires_python


def test_release_requires_python_picks_first_non_empty() -> None:
    pkg = Package.from_json(
        _payload({
            "1.0.0": [
                {"requires_python": ""},
                {"requires_python": ">=3.10"},
                {"requires_python": ">=3.11"},
            ]
        })
    )
    release = pkg.get_release("1.0.0")
    assert release is not None
    assert release.requires_python == SpecifierSet(">=3.10")


@pytest.mark.parametrize("raw", [None, "", "not a specifier"])
def test_release_requires_python_none_for_missing_or_invalid(raw: str | None) -> None:
    payload_dist = {} if raw is None else {"requires_python": raw}
    pkg = Package.from_json(_payload({"1.0.0": [payload_dist]}))
    release = pkg.get_release("1.0.0")
    assert release is not None
    assert release.requires_python is None
