from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest
from pypi import Package

from dev import update_ml_package_versions
from dev.update_ml_package_versions import VersionInfo


def _iso8601(dt: datetime) -> str:
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()


def package_from_version_infos(version_infos: list[VersionInfo]) -> Package:
    return Package.from_json({
        "releases": {
            v.version: [
                {
                    "filename": v.version + ".whl",
                    "upload_time_iso_8601": _iso8601(v.upload_time),
                }
            ]
            for v in version_infos
        },
    })


def package_from_versions(versions: list[str]) -> Package:
    fixed = datetime(2023, 10, 4, 16, 38, 57, tzinfo=timezone.utc)
    return package_from_version_infos([VersionInfo(v, fixed) for v in versions])


@pytest.fixture(autouse=True)
def change_working_directory(tmp_path, monkeypatch):
    """
    Changes the current working directory to a temporary directory to avoid modifying files in the
    repository.
    """
    monkeypatch.chdir(tmp_path)


def run_test(src, src_expected, mock_packages):
    async def fake_get_packages(names):
        return [mock_packages[n] for n in names]

    versions_yaml = Path("mlflow/ml-package-versions.yml")
    versions_yaml.parent.mkdir()
    versions_yaml.write_text(src)

    with (
        mock.patch("dev.update_ml_package_versions.get_packages", new=fake_get_packages),
        mock.patch("dev.update_ml_package_versions.check_pypi_accessibility") as mock_check,
    ):
        update_ml_package_versions.update()
        mock_check.assert_called_once()

    assert versions_yaml.read_text() == src_expected


def test_multiple_flavors_are_correctly_updated():
    src = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    maximum: "0.0.1"
xgboost:
  package_info:
    pip_release: xgboost
  autologging:
    maximum: "0.1.1"
"""
    mock_packages = {
        "sklearn": package_from_versions(["0.0.2"]),
        "xgboost": package_from_versions(["0.1.2"]),
    }
    src_expected = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    maximum: "0.0.2"
xgboost:
  package_info:
    pip_release: xgboost
  autologging:
    maximum: "0.1.2"
"""
    run_test(src, src_expected, mock_packages)


def test_both_models_and_autologging_are_updated():
    src = """
sklearn:
  package_info:
    pip_release: sklearn
  models:
    maximum: "0.0.1"
  autologging:
    maximum: "0.0.1"
"""
    mock_packages = {
        "sklearn": package_from_versions(["0.0.2"]),
    }
    src_expected = """
sklearn:
  package_info:
    pip_release: sklearn
  models:
    maximum: "0.0.2"
  autologging:
    maximum: "0.0.2"
"""
    run_test(src, src_expected, mock_packages)


def test_pre_and_dev_versions_are_ignored():
    src = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    maximum: "0.0.1"
"""
    mock_packages = {
        "sklearn": package_from_versions([
            # pre-release and dev-release should be filtered out
            "0.0.3.rc1",  # pre-release
            "0.0.3.dev1",  # dev-release
            "0.0.2.post",  # post-release
            "0.0.2",  # final release
        ]),
    }
    src_expected = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    maximum: "0.0.2.post0"
"""
    run_test(src, src_expected, mock_packages)


def test_unsupported_versions_are_ignored():
    src = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    unsupported: ["0.0.3"]
    maximum: "0.0.1"
"""
    mock_packages = {"sklearn": package_from_versions(["0.0.2", "0.0.3"])}
    src_expected = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    unsupported: ["0.0.3"]
    maximum: "0.0.2"
"""
    run_test(src, src_expected, mock_packages)


def test_freeze_field_prevents_updating_maximum_version():
    src = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    pin_maximum: True
    maximum: "0.0.1"
"""
    mock_packages = {"sklearn": package_from_versions(["0.0.2"])}
    src_expected = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    pin_maximum: True
    maximum: "0.0.1"
"""
    run_test(src, src_expected, mock_packages)


def test_update_min_supported_version():
    src = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    minimum: "0.0.1"
    maximum: "0.0.8"
"""
    mock_packages = {
        "sklearn": package_from_version_infos([
            VersionInfo("0.0.2", datetime.now() - timedelta(days=1000)),
            VersionInfo("0.0.3", datetime.now() - timedelta(days=365)),
            VersionInfo("0.0.8", datetime.now() - timedelta(days=180)),
        ])
    }
    src_expected = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    minimum: "0.0.3"
    maximum: "0.0.8"
"""
    run_test(src, src_expected, mock_packages)


def test_update_min_supported_version_for_dead_package():
    src = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    minimum: "0.0.7"
    maximum: "0.0.8"
"""
    mock_packages = {
        "sklearn": package_from_version_infos([
            VersionInfo("0.0.7", datetime.now() - timedelta(days=1000)),
            VersionInfo("0.0.8", datetime.now() - timedelta(days=800)),
        ])
    }
    src_expected = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    minimum: "0.0.8"
    maximum: "0.0.8"
"""
    run_test(src, src_expected, mock_packages)
