import pytest

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
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
        ("other://user:p%40ss@host:123/db?x=1", "other://***@host:123/db"),
        ("other://user:password@[::1]:123/db", "other://***@[::1]:123/db"),
        ("other://user:password@extra@host/db", "other://***@host/db"),
        ("other://user@host/db", "other://***@host/db"),
        ("other://user:password@[broken/db", "<redacted URI>"),
        ("other://user:pa/ss@host/db", "<redacted URI>"),
        ("other://user:pa?ss@host/db", "<redacted URI>"),
        ("other://user:pa#ss@host/db", "<redacted URI>"),
        ("other://host/db?password=secret", "other://host/db"),
        ("other://host/db#secret", "other://host/db"),
        ("other://host/db?odbc_connect=UID%3Duser%3BPWD%3Dsecret", "other://host/db"),
        ("other:///?odbc_connect=PWD%3Dsecret", "other:/"),
        ("other://host/db;password=secret", "other://host/db"),
        ("other:///db;password=secret", "other:/db"),
        ("other://host/db?x=1", "other://host/db"),
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
    assert "secret" not in str(error)
    assert error.supported_uri_schemes == ["file", "sqlite"]


@pytest.mark.parametrize("method", ["search_registered_models", "list_workspaces"])
def test_client_unsupported_store_uri_error_redacts_credentials(method):
    uri = "other://user:password@host/db?password=secret"
    client = MlflowClient(
        tracking_uri="http://localhost:5000", registry_uri=uri, workspace_store_uri=uri
    )
    with pytest.raises(MlflowException, match=r"other://\*\*\*@host/db") as exc:
        getattr(client, method)()
    assert "user:password" not in str(exc.value)
    assert "secret" not in str(exc.value)
    assert exc.value.error_code == "FEATURE_DISABLED"
