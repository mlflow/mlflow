import importlib.metadata
from unittest import mock

import pytest

from mlflow.store.artifact.databricks_sdk_artifact_repo import _sdk_supports_large_file_uploads


def test_sdk_supports_large_file_uploads_true():
    with mock.patch("importlib.metadata.version", return_value="0.45.0") as mock_version:
        assert _sdk_supports_large_file_uploads() is True
        mock_version.assert_called_once_with("databricks-sdk")


def test_sdk_supports_large_file_uploads_false_for_old_version():
    with mock.patch("importlib.metadata.version", return_value="0.44.0"):
        assert _sdk_supports_large_file_uploads() is False


@pytest.mark.parametrize(
    "side_effect_or_return",
    [
        {"side_effect": importlib.metadata.PackageNotFoundError("databricks-sdk")},
        {"return_value": None},
    ],
)
def test_sdk_supports_large_file_uploads_missing_version_does_not_raise(side_effect_or_return):
    # On Databricks Serverless the vendored databricks-sdk can report no version, which previously
    # crashed `mlflow.log_model()` with `TypeError` from `Version(None)`. It must now degrade to
    # `False` instead.
    with mock.patch("importlib.metadata.version", **side_effect_or_return):
        assert _sdk_supports_large_file_uploads() is False
