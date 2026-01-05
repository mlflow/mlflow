import os
import sys

import pytest


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
def test_skinny_client_without_importing_sqlalchemy() -> None:
    from mlflow.tracking.client import MlflowClient

    client = MlflowClient(
        tracking_uri="databricks",
        registry_uri="databricks",
    )
    client._tracking_client.store
    client._get_registry_client().store
    assert "sqlalchemy" not in sys.modules
