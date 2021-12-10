import json
import os
import re
from unittest import mock
import tempfile

from dev import update_ml_package_versions


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
        return cls({"releases": {v: [v + ".whl"] for v in versions}})


def run_test(src, src_expected, mock_responses):
    def patch_urlopen(url):
        package_name = re.search(r"https://pypi.python.org/pypi/(.+)/json", url).group(1)
        return mock_responses[package_name]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "versions.yml")
        with open(tmp_path, "w") as f:
            f.write(src)

        with mock.patch("urllib.request.urlopen", new=patch_urlopen):
            update_ml_package_versions.main(["--path", tmp_path])

        with open(tmp_path) as f:
            assert f.read() == src_expected


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
