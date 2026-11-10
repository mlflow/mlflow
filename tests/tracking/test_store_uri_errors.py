import pytest

from mlflow.genai.labeling.stores import UnsupportedLabelingStoreURIException
from mlflow.genai.scorers.registry import UnsupportedScorerStoreURIException
from mlflow.tracking._workspace.registry import UnsupportedWorkspaceStoreURIException
from mlflow.tracking.registry import UnsupportedModelRegistryStoreURIException


@pytest.mark.parametrize(
    "exception",
    [
        UnsupportedModelRegistryStoreURIException,
        UnsupportedWorkspaceStoreURIException,
        UnsupportedLabelingStoreURIException,
        UnsupportedScorerStoreURIException,
    ],
)
@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("mongodb+srv://user:password@host/db", "mongodb+srv://***@host/db"),
        ("other://user:p%40ss@host:123/db?x=1", "other://***@host:123/db?x=1"),
        ("other://user:password@[::1]:123/db", "other://***@[::1]:123/db"),
        ("other://user:password@extra@host/db", "other://***@host/db"),
        ("other://user@host/db", "other://***@host/db"),
        ("other://user:password@[broken/db", "<redacted URI>"),
        ("other://host/db?x=1", "other://host/db?x=1"),
        ("databricks", "databricks"),
        ("databricks-uc", "databricks-uc"),
        ("sqlite:///store.db", "sqlite:///store.db"),
        ("/tmp/store@host", "/tmp/store@host"),
        (r"C:\store", r"C:\store"),
        ("", ""),
        (None, None),
    ],
)
def test_unsupported_store_uri_error_redacts_credentials(exception, uri, expected):
    error = exception(uri, ["file", "sqlite"])
    assert f"'{expected}'" in str(error)
    assert "user:password" not in str(error)
    assert error.supported_uri_schemes == ["file", "sqlite"]
