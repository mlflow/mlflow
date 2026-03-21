"""Tests for the SDK auth probe caching optimisation.

Verifies that ``probe_databricks_sdk_auth`` runs once at store
construction time, and that subsequent calls to
``_get_databricks_host_creds_impl`` with a pre-computed
``_use_databricks_sdk`` value skip the probe ``WorkspaceClient``.
"""

from unittest.mock import MagicMock, patch

import pytest

_HOST = "https://test.cloud.databricks.com"


def _mock_config():
    return MagicMock(
        host=_HOST,
        username=None,
        password=None,
        token=None,
        insecure=False,
        client_id=None,
        client_secret=None,
    )


@pytest.fixture(autouse=True)
def _enable_db_sdk(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_DB_SDK", "true")


@pytest.fixture
def mock_ws_client():
    with patch("databricks.sdk.WorkspaceClient") as mock_cls:
        mock_cls.return_value = MagicMock()
        yield mock_cls


# -- probe_databricks_sdk_auth ------------------------------------------------


def test_probe_returns_true_when_sdk_auth_succeeds(mock_ws_client):
    from mlflow.utils.databricks_utils import probe_databricks_sdk_auth

    assert probe_databricks_sdk_auth("databricks") is True
    assert mock_ws_client.call_count == 1


def test_probe_returns_false_when_sdk_auth_fails(mock_ws_client):
    mock_ws_client.side_effect = Exception("no creds")
    from mlflow.utils.databricks_utils import probe_databricks_sdk_auth

    assert probe_databricks_sdk_auth("databricks") is False


def test_probe_returns_false_when_db_sdk_disabled(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_DB_SDK", "false")
    from mlflow.utils.databricks_utils import probe_databricks_sdk_auth

    assert probe_databricks_sdk_auth("databricks") is False


# -- _get_databricks_host_creds_impl: pre-computed vs legacy -------------------


def test_impl_skips_probe_when_precomputed_true(mock_ws_client):
    from mlflow.utils.databricks_utils import _get_databricks_host_creds_impl

    with patch(
        "mlflow.utils.databricks_utils._get_databricks_creds_config",
        return_value=_mock_config(),
    ):
        creds = _get_databricks_host_creds_impl("databricks", _use_databricks_sdk=True)

    assert creds.use_databricks_sdk is True
    assert mock_ws_client.call_count == 0


def test_impl_skips_probe_when_precomputed_false(mock_ws_client):
    from mlflow.utils.databricks_utils import _get_databricks_host_creds_impl

    with patch(
        "mlflow.utils.databricks_utils._get_databricks_creds_config",
        return_value=_mock_config(),
    ):
        creds = _get_databricks_host_creds_impl("databricks", _use_databricks_sdk=False)

    assert creds.use_databricks_sdk is False
    assert mock_ws_client.call_count == 0


def test_impl_probes_every_call_when_none(mock_ws_client):
    from mlflow.utils.databricks_utils import _get_databricks_host_creds_impl

    with patch(
        "mlflow.utils.databricks_utils._get_databricks_creds_config",
        return_value=_mock_config(),
    ):
        for _ in range(5):
            _get_databricks_host_creds_impl("databricks", _use_databricks_sdk=None)

    assert mock_ws_client.call_count == 5


# -- get_databricks_host_creds: backward compatibility -------------------------


def test_public_api_still_probes_every_call(mock_ws_client):
    from mlflow.utils.databricks_utils import get_databricks_host_creds

    with patch(
        "mlflow.utils.databricks_utils._get_databricks_creds_config",
        return_value=_mock_config(),
    ):
        for _ in range(5):
            get_databricks_host_creds("databricks")

    assert mock_ws_client.call_count == 5


# -- Regression: store's get_host_creds callback probes only once --------------


def test_databricks_rest_store_probes_once_for_many_requests(mock_ws_client):
    from mlflow.tracking._tracking_service.utils import _get_databricks_rest_store

    with patch(
        "mlflow.utils.databricks_utils._get_databricks_creds_config",
        return_value=_mock_config(),
    ):
        # 1 probe during store construction.
        store = _get_databricks_rest_store("databricks")

        # 50 simulated REST requests via the store's get_host_creds callback.
        for _ in range(50):
            store.get_host_creds()

    # Before the fix this was 51 (1 in factory + 50 in get_host_creds).
    # After the fix it is 1 (only the factory probe).
    assert mock_ws_client.call_count == 1
