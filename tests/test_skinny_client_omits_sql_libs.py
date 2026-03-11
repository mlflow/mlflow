import os
import sys

import pytest

from mlflow.exceptions import MlflowException
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.tracking.client import MlflowClient


@pytest.mark.skipif(
    "MLFLOW_SKINNY" not in os.environ, reason="This test is only valid for the skinny client"
)
def test_fails_import_sqlalchemy():
    import mlflow  # noqa: F401

    with pytest.raises(ImportError, match="sqlalchemy"):
        import sqlalchemy  # noqa: F401


@pytest.mark.skipif(
    "MLFLOW_SKINNY" not in os.environ, reason="This test is only valid for the skinny client"
)
def test_skinny_client_sqlite_uri_gives_helpful_error() -> None:
    with pytest.raises(MlflowException, match="requires the 'sqlalchemy' and 'alembic' packages"):
        _get_store("sqlite:///mlflow.db")


@pytest.mark.skipif(
    "MLFLOW_SKINNY" not in os.environ, reason="This test is only valid for the skinny client"
)
def test_skinny_client_without_importing_sqlalchemy() -> None:
    client = MlflowClient(
        tracking_uri="databricks",
        registry_uri="databricks",
    )
    client._tracking_client.store
    client._get_registry_client().store
    assert "sqlalchemy" not in sys.modules
