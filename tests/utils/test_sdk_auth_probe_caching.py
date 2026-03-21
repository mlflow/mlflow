"""Tests for the SDK auth probe caching optimisation.

Background
----------
``WorkspaceClient.__init__`` performs SDK auth initialisation for a URI:
it reads env vars, fetches ``/.well-known/databricks-config``, walks the
auth chain (pat → basic → metadata-service → oauth-m2m), and fetches the
OIDC discovery document.  This is expensive (2-4 HTTPS round-trips with
``oauth-m2m``).

Before this fix, every call to ``get_host_creds()`` — which happens on
every MLflow REST request — re-ran this initialisation by constructing a
throwaway ``WorkspaceClient``.  After the fix, the probe runs once at
store construction time and the result is reused for all subsequent
requests.

These tests verify:
1. ``should_use_databricks_sdk`` correctly probes once and returns a bool.
2. ``_get_databricks_host_creds_impl`` skips the probe when given a
   pre-computed value.
3. ``get_databricks_host_creds`` preserves backward-compat (still probes).
4. The store's ``get_host_creds`` callback makes no network calls after
   store construction.
5. ``WorkspaceClient.__init__`` (SDK auth initialisation) is called once
   per store, not once per request.
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


# -- should_use_databricks_sdk -------------------------------------------------


def test_probe_returns_true_when_sdk_auth_succeeds(mock_ws_client):
    from mlflow.utils.databricks_utils import should_use_databricks_sdk

    assert should_use_databricks_sdk("databricks") is True
    assert mock_ws_client.call_count == 1


def test_probe_returns_false_when_sdk_auth_fails(mock_ws_client):
    mock_ws_client.side_effect = Exception("no creds")
    from mlflow.utils.databricks_utils import should_use_databricks_sdk

    assert should_use_databricks_sdk("databricks") is False


def test_probe_returns_false_when_db_sdk_disabled(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_DB_SDK", "false")
    from mlflow.utils.databricks_utils import should_use_databricks_sdk

    assert should_use_databricks_sdk("databricks") is False


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


# -- Store-level: get_host_creds makes no network calls after construction -----


def test_get_host_creds_makes_no_http_calls_after_store_construction(mock_ws_client):
    """After a Databricks store is constructed, its get_host_creds callback
    should return credentials without making any HTTP requests.  This is
    the user-facing contract: REST request latency is not inflated by
    redundant OIDC discovery round-trips.
    """
    from mlflow.tracking._tracking_service.utils import _get_databricks_rest_store

    with patch(
        "mlflow.utils.databricks_utils._get_databricks_creds_config",
        return_value=_mock_config(),
    ):
        store = _get_databricks_rest_store("databricks")

        # Track all HTTP calls made after construction.
        with patch("urllib3.HTTPSConnectionPool.urlopen") as mock_urlopen:
            for _ in range(50):
                store.get_host_creds()

        # No HTTP calls — all auth discovery happened during construction.
        assert mock_urlopen.call_count == 0


def test_store_construction_initialises_sdk_auth_once(mock_ws_client):
    """WorkspaceClient.__init__ performs SDK auth initialisation for a URI
    (env-var parsing, OIDC discovery, auth-chain walk).  Before this fix,
    this happened on every REST request because each get_host_creds call
    constructed a throwaway WorkspaceClient.  After the fix, it happens
    once during store construction.

    This test asserts that WorkspaceClient is constructed exactly once for
    the lifetime of a store, regardless of how many requests are made.
    """
    from mlflow.tracking._tracking_service.utils import _get_databricks_rest_store

    with patch(
        "mlflow.utils.databricks_utils._get_databricks_creds_config",
        return_value=_mock_config(),
    ):
        store = _get_databricks_rest_store("databricks")

        for _ in range(50):
            store.get_host_creds()

    # 1 construction during store creation, 0 during get_host_creds calls.
    # Before the fix this was 51 (1 + 50).
    assert mock_ws_client.call_count == 1
