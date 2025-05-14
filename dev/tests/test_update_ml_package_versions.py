import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest

from dev import update_ml_package_versions
from dev.update_ml_package_versions import VersionInfo


class MockResponse:
    def __init__(self, body):
        self.body = json.dumps(body).encode("utf-8")

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @classmethod
    def from_versions(cls, versions):
        return cls(
            {
                "releases": {
                    v: [
                        {
                            "filename": v + ".whl",
                            "upload_time": "2023-10-04T16:38:57",
                        }
                    ]
                    for v in versions
                }
            }
        )

    @classmethod
    def from_version_infos(cls, version_infos: list[VersionInfo]) -> "MockResponse":
        return cls(
            {
                "releases": {
                    v.version: [
                        {
                            "filename": v.version + ".whl",
                            "upload_time": v.upload_time.isoformat(),
                        }
                    ]
                    for v in version_infos
                }
            }
        )


@pytest.fixture(autouse=True)
def change_working_directory(tmp_path, monkeypatch):
    """
    Changes the current working directory to a temporary directory to avoid modifying files in the
    repository.
    """
    monkeypatch.chdir(tmp_path)


def run_test(src, src_expected, mock_responses):
    def patch_urlopen(url):
        package_name = re.search(r"https://pypi.python.org/pypi/(.+)/json", url).group(1)
        return mock_responses[package_name]

    versions_yaml = Path("mlflow/ml-package-versions.yml")
    versions_yaml.parent.mkdir()
    versions_yaml.write_text(src)

    with mock.patch("urllib.request.urlopen", new=patch_urlopen):
        update_ml_package_versions.update()

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
    mock_responses = {
        "sklearn": MockResponse.from_versions(["0.0.2"]),
        "xgboost": MockResponse.from_versions(["0.1.2"]),
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
    run_test(src, src_expected, mock_responses)


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
    mock_responses = {
        "sklearn": MockResponse.from_versions(["0.0.2"]),
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
    run_test(src, src_expected, mock_responses)


def test_pre_and_dev_versions_are_ignored():
    src = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    maximum: "0.0.1"
"""
    mock_responses = {
        "sklearn": MockResponse.from_versions(
            [
                # pre-release and dev-release should be filtered out
                "0.0.3.rc1",  # pre-release
                "0.0.3.dev1",  # dev-release
                "0.0.2.post",  # post-release
                "0.0.2",  # final release
            ]
        ),
    }
    src_expected = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    maximum: "0.0.2.post"
"""
    run_test(src, src_expected, mock_responses)


def test_unsupported_versions_are_ignored():
    src = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    unsupported: ["0.0.3"]
    maximum: "0.0.1"
"""
    mock_responses = {"sklearn": MockResponse.from_versions(["0.0.2", "0.0.3"])}
    src_expected = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    unsupported: ["0.0.3"]
    maximum: "0.0.2"
"""
    run_test(src, src_expected, mock_responses)


def test_freeze_field_prevents_updating_maximum_version():
    src = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    pin_maximum: True
    maximum: "0.0.1"
"""
    mock_responses = {"sklearn": MockResponse.from_versions(["0.0.2"])}
    src_expected = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    pin_maximum: True
    maximum: "0.0.1"
"""
    run_test(src, src_expected, mock_responses)


def test_update_min_supported_version():
    src = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    minimum: "0.0.1"
    maximum: "0.0.8"
"""
    mock_responses = {
        "sklearn": MockResponse.from_version_infos(
            [
                VersionInfo("0.0.2", datetime.now() - timedelta(days=1000)),
                VersionInfo("0.0.3", datetime.now() - timedelta(days=365)),
                VersionInfo("0.0.8", datetime.now() - timedelta(days=180)),
            ]
        )
    }
    src_expected = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    minimum: "0.0.3"
    maximum: "0.0.8"
"""
    run_test(src, src_expected, mock_responses)


def test_update_min_supported_version_for_dead_package():
    src = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    minimum: "0.0.7"
    maximum: "0.0.8"
"""
    mock_responses = {
        "sklearn": MockResponse.from_version_infos(
            [
                VersionInfo("0.0.7", datetime.now() - timedelta(days=1000)),
                VersionInfo("0.0.8", datetime.now() - timedelta(days=800)),
            ]
        )
    }
    src_expected = """
sklearn:
  package_info:
    pip_release: sklearn
  autologging:
    minimum: "0.0.8"
    maximum: "0.0.8"
"""
    run_test(src, src_expected, mock_responses)
