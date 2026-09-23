import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest

from mlflow.server.assistant import gateway_permissions as gp


def _endpoint(name: str, endpoint_id: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, endpoint_id=endpoint_id)


@pytest.fixture
def auth_store(monkeypatch):
    store = mock.MagicMock()
    store.list_gateway_endpoint_permissions.return_value = []

    # Mirror the real get_permission, which indexes a fixed table and raises KeyError on an
    # unknown/legacy permission name.
    can_use_by_name = {"READ": False, "USE": True, "EDIT": True, "MANAGE": True}

    auth_module = types.ModuleType("mlflow.server.auth")
    auth_module.store = store
    perms_module = types.ModuleType("mlflow.server.auth.permissions")
    perms_module.USE = SimpleNamespace(name="USE")
    perms_module.get_permission = lambda name: SimpleNamespace(can_use=can_use_by_name[name])
    monkeypatch.setitem(sys.modules, "mlflow.server.auth", auth_module)
    monkeypatch.setitem(sys.modules, "mlflow.server.auth.permissions", perms_module)
    return store


def _patch_endpoints(monkeypatch, endpoints):
    tracking_store = mock.MagicMock()
    tracking_store.list_gateway_endpoints.return_value = endpoints
    monkeypatch.setattr(
        "mlflow.tracking._tracking_service.utils._get_store", lambda: tracking_store
    )


def test_noop_without_auth_plugin(monkeypatch, auth_store):
    monkeypatch.setattr(gp, "auth_plugin_active", lambda: False)
    _patch_endpoints(monkeypatch, [_endpoint("mlflow-assistant-openai", "e1")])

    gp.ensure_assistant_gateway_use_permission("alice")

    auth_store.grant_user_permission.assert_not_called()


def test_noop_without_username(monkeypatch, auth_store):
    monkeypatch.setattr(gp, "auth_plugin_active", lambda: True)

    gp.ensure_assistant_gateway_use_permission(None)

    auth_store.grant_user_permission.assert_not_called()


def test_grants_use_on_managed_endpoints_only(monkeypatch, auth_store):
    monkeypatch.setattr(gp, "auth_plugin_active", lambda: True)
    _patch_endpoints(
        monkeypatch,
        [
            _endpoint("mlflow-assistant-openai", "e1"),
            # An operator's own endpoint is never auto-granted (that would escalate privilege) --
            # neither an unrelated name nor one that merely shares the prefix but is not an exact
            # managed name.
            _endpoint("team-shared-openai", "e2"),
            _endpoint("mlflow-assistant-custom", "e3"),
        ],
    )

    gp.ensure_assistant_gateway_use_permission("alice")

    auth_store.grant_user_permission.assert_called_once_with(
        "alice", "gateway_endpoint", "e1", "USE"
    )


def test_grants_across_multiple_managed_endpoints(monkeypatch, auth_store):
    monkeypatch.setattr(gp, "auth_plugin_active", lambda: True)
    _patch_endpoints(
        monkeypatch,
        [
            _endpoint("mlflow-assistant-openai", "e1"),
            _endpoint("mlflow-assistant-anthropic", "e2"),
        ],
    )

    gp.ensure_assistant_gateway_use_permission("alice")

    granted = {c.args[2] for c in auth_store.grant_user_permission.call_args_list}
    assert granted == {"e1", "e2"}


def test_no_gateway_support_is_a_quiet_noop(monkeypatch, auth_store):
    monkeypatch.setattr(gp, "auth_plugin_active", lambda: True)
    tracking_store = mock.MagicMock()
    tracking_store.list_gateway_endpoints.side_effect = NotImplementedError("FileStore")
    monkeypatch.setattr(
        "mlflow.tracking._tracking_service.utils._get_store", lambda: tracking_store
    )

    gp.ensure_assistant_gateway_use_permission("alice")

    auth_store.grant_user_permission.assert_not_called()


def test_permission_read_failure_does_not_break_the_turn(monkeypatch, auth_store):
    # A failure reading existing permissions (e.g. a legacy permission name get_permission does
    # not know, or a workspace/store error) must fail open, not raise into the SSE stream.
    monkeypatch.setattr(gp, "auth_plugin_active", lambda: True)
    auth_store.list_gateway_endpoint_permissions.return_value = [
        SimpleNamespace(endpoint_id="e1", permission="LEGACY_UNKNOWN")
    ]
    _patch_endpoints(monkeypatch, [_endpoint("mlflow-assistant-openai", "e1")])

    gp.ensure_assistant_gateway_use_permission("alice")  # must not raise

    auth_store.grant_user_permission.assert_not_called()


def test_skips_endpoints_the_user_can_already_use(monkeypatch, auth_store):
    monkeypatch.setattr(gp, "auth_plugin_active", lambda: True)
    auth_store.list_gateway_endpoint_permissions.return_value = [
        SimpleNamespace(endpoint_id="e1", permission="USE")
    ]
    _patch_endpoints(monkeypatch, [_endpoint("mlflow-assistant-openai", "e1")])

    gp.ensure_assistant_gateway_use_permission("alice")

    auth_store.grant_user_permission.assert_not_called()


def test_grant_failure_does_not_raise(monkeypatch, auth_store):
    # A failed grant must not break the turn; the gateway's USE check remains authoritative.
    monkeypatch.setattr(gp, "auth_plugin_active", lambda: True)
    auth_store.grant_user_permission.side_effect = RuntimeError("db down")
    _patch_endpoints(monkeypatch, [_endpoint("mlflow-assistant-openai", "e1")])

    gp.ensure_assistant_gateway_use_permission("alice")

    auth_store.grant_user_permission.assert_called_once()
