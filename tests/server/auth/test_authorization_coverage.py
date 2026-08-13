# CI guard for the basic-auth dispatcher: every Flask route must resolve to an
# authorization decision, or be listed in the debt list _KNOWN_UNGATED_ROUTE_MARKERS.

import json
from types import SimpleNamespace

from mlflow.server import auth as a
from mlflow.server.handlers import get_endpoints


class _Req:
    def __init__(self, path, method):
        self.path = path
        self.method = method


def _is_covered(path, method):
    req = _Req(path, method)
    return (
        a.is_unprotected_route(path)
        or a._find_validator(req) is not None
        or a._is_proxy_artifact_path(path)
        or a._authorized_outside_before_request(req)
    )


def _all_routes():
    seen = set()
    for path, handler, methods in get_endpoints():
        if getattr(handler, "__name__", str(handler)) == "_not_implemented":
            continue
        for method in methods:
            if (path, method) not in seen:
                seen.add((path, method))
                yield path, method


def test_no_new_ungated_routes():
    uncovered = sorted(
        f"{method:6} {path}"
        for path, method in _all_routes()
        if not _is_covered(path, method) and not a._is_known_ungated_route(path)
    )
    assert not uncovered, (
        "Routes with no authorization decision that are not in the documented debt "
        "list (_KNOWN_UNGATED_ROUTE_MARKERS). Add a before-request validator, an "
        "after-request filter, a self-authorizing handler + allowlist entry, or an "
        "explicit public-route entry:\n" + "\n".join(uncovered)
    )


def test_all_dataset_routes_resolve_a_real_validator():
    # Guard against a dataset route silently hitting the hard-deny lambda fallback in
    # _find_validator (which test_no_new_ungated_routes would still count as gated).
    real = set(a.DATASET_BEFORE_REQUEST_HANDLERS.values()) | set(
        a.DATASET_EXACT_BEFORE_REQUEST_HANDLERS.values()
    )
    unresolved = sorted(
        f"{method:6} {path}"
        for path, method in _all_routes()
        if "/mlflow/datasets/" in path and a._find_validator(_Req(path, method)) not in real
    )
    assert not unresolved, (
        "Dataset routes that fall through to the hard-deny fallback instead of a real "
        "validator (add them to DATASET_BEFORE_REQUEST_HANDLERS):\n" + "\n".join(unresolved)
    )


def test_known_ungated_markers_are_not_stale():
    matched = set()
    for path, method in _all_routes():
        if _is_covered(path, method):
            continue
        for marker in a._KNOWN_UNGATED_ROUTE_MARKERS:
            if marker in path:
                matched.add(marker)
    stale = [m for m in a._KNOWN_UNGATED_ROUTE_MARKERS if m not in matched]
    assert not stale, (
        "Debt markers that no longer match any ungated route (their family is now "
        f"gated — remove them from _KNOWN_UNGATED_ROUTE_MARKERS): {stale}"
    )


def test_unknown_issue_subpath_fails_closed():
    # An unrecognized /mlflow/issues/ path resolves to a denying validator (not None), so
    # it is denied even with MLFLOW_BASIC_AUTH_FAIL_CLOSED off, matching the dataset/trace
    # branches, rather than silently falling through.
    v = a._find_validator(_Req("/api/3.0/mlflow/issues/i-1/comments", "POST"))
    assert v is not None
    assert v() is False
    # The invoke route is exempt from that hard-deny: it either has its own validator or is
    # still tracked in the debt list, never the fail-closed fallback.
    invoke = "/ajax-api/3.0/mlflow/issues/invoke"
    assert a._is_known_ungated_route(invoke) or (invoke, "POST") in a.BEFORE_REQUEST_VALIDATORS


def _fastapi_native_routes():
    # Routers served directly by FastAPI (Flask is mounted separately and authorized by
    # _before_request, so it is out of scope for this guard).
    from mlflow.server.fastapi_app import (
        artifact_router,
        assistant_router,
        gateway_router,
        job_api_router,
        mcp_server_router,
        otel_router,
    )

    def _methods(route):
        return [
            m
            for m in sorted(getattr(route, "methods", None) or ["GET"])
            if m not in ("HEAD", "OPTIONS")
        ]

    for router in (artifact_router, assistant_router, gateway_router, job_api_router, otel_router):
        for route in router.routes:
            for method in _methods(route):
                yield route.path, method
    # The MCP router is mounted under versioned prefixes; prepend one to form full paths.
    mcp_prefix = next(iter(a.get_mcp_server_api_route_prefixes())).rstrip("/")
    for route in mcp_server_router.routes:
        for method in _methods(route):
            yield mcp_prefix + route.path, method


def test_no_ungated_fastapi_native_routes():
    # The FastAPI permission middleware is fail-open for routes with no validator; this
    # guard makes a new native route without one a failing test rather than a silent hole.
    uncovered = sorted(
        f"{method:6} {path}"
        for path, method in _fastapi_native_routes()
        if a._find_fastapi_validator(path, method) is None
        and not a.is_unprotected_route(path)
        and not any(m in path for m in a._KNOWN_UNGATED_FASTAPI_ROUTE_MARKERS)
    )
    assert not uncovered, (
        "Native FastAPI routes with no authorization validator that are not documented in "
        "_KNOWN_UNGATED_FASTAPI_ROUTE_MARKERS. Add a validator branch in "
        "_find_fastapi_validator, or document the exemption:\n" + "\n".join(uncovered)
    )


def test_jobs_owner_check(monkeypatch):
    # Jobs get/cancel are gated on ownership — the recorded creator must match the
    # caller; a different creator or no creator is denied (admins bypass upstream).
    import mlflow.server.jobs as jobs_mod

    monkeypatch.setattr(a, "authenticate_request", lambda: SimpleNamespace(username="alice"))
    monkeypatch.setattr(a, "_get_request_param", lambda p: "job-1")

    monkeypatch.setattr(jobs_mod, "get_job", lambda jid: SimpleNamespace(creator="alice"))
    assert a.validate_is_job_owner() is True

    monkeypatch.setattr(jobs_mod, "get_job", lambda jid: SimpleNamespace(creator="bob"))
    assert a.validate_is_job_owner() is False

    monkeypatch.setattr(jobs_mod, "get_job", lambda jid: SimpleNamespace(creator=None))
    assert a.validate_is_job_owner() is False


def test_gateway_discovery_config_budget_gating():
    def vname(path, method="GET"):
        v = a._find_validator(_Req(path, method))
        return v.__name__ if v is not None else None

    # Discovery + config back the gateway UI for any signed-in user (no secret material).
    assert vname("/ajax-api/3.0/mlflow/gateway/supported-providers") == "_allow_authenticated"
    assert vname("/ajax-api/3.0/mlflow/gateway/supported-models") == "_allow_authenticated"
    assert vname("/ajax-api/3.0/mlflow/gateway/provider-config") == "_allow_authenticated"
    assert vname("/ajax-api/3.0/mlflow/gateway/secrets/config") == "_allow_authenticated"
    # Budget-policy reads are authenticated-open; writes are admin-only (matches #21120).
    assert vname("/api/3.0/mlflow/gateway/budgets/get") == "_allow_authenticated"
    assert vname("/api/3.0/mlflow/gateway/budgets/list") == "_allow_authenticated"
    assert vname("/api/3.0/mlflow/gateway/budgets/windows") == "_allow_authenticated"
    assert vname("/api/3.0/mlflow/gateway/budgets/create", "POST") == "sender_is_admin"
    assert vname("/api/3.0/mlflow/gateway/budgets/delete", "DELETE") == "sender_is_admin"


def test_gateway_guardrail_gating():
    def vname(path, method):
        v = a._find_validator(_Req(path, method))
        return v.__name__ if v is not None else None

    base = "/api/3.0/mlflow/gateway/guardrails"
    # Standalone guardrail CRUD -> admin-only.
    assert vname(f"{base}/create", "POST") == "sender_is_admin"
    assert vname(f"{base}/get", "GET") == "sender_is_admin"
    assert vname(f"{base}/list", "GET") == "sender_is_admin"
    assert vname(f"{base}/delete", "DELETE") == "sender_is_admin"
    # Endpoint-attached routes gate on the owning gateway endpoint.
    assert vname(f"{base}/add-to-endpoint", "POST") == "validate_can_update_gateway_endpoint"
    assert vname(f"{base}/remove-from-endpoint", "DELETE") == "validate_can_update_gateway_endpoint"
    assert vname(f"{base}/update-config", "PATCH") == "validate_can_update_gateway_endpoint"
    assert vname(f"{base}/list-for-endpoint", "GET") == "validate_can_read_gateway_endpoint"


def test_filter_list_gateway_endpoints_drops_unreadable(monkeypatch):
    # Cross-resource gateway list endpoints are filtered after-request to rows the
    # caller can read.
    monkeypatch.setattr(a, "sender_is_admin", lambda: False)
    monkeypatch.setattr(a, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        a, "_role_based_read_predicate", lambda username, rt: lambda rid: rid == "ep-allowed"
    )
    resp = SimpleNamespace(
        json={"endpoints": [{"endpoint_id": "ep-allowed"}, {"endpoint_id": "ep-denied"}]},
        data=None,
    )
    a.filter_list_gateway_endpoints(resp)
    ids = [e["endpoint_id"] for e in json.loads(resp.data)["endpoints"]]
    assert ids == ["ep-allowed"]


def test_filter_list_gateway_model_definitions_drops_unreadable(monkeypatch):
    monkeypatch.setattr(a, "sender_is_admin", lambda: False)
    monkeypatch.setattr(a, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        a, "_role_based_read_predicate", lambda username, rt: lambda rid: rid == "md-allowed"
    )
    resp = SimpleNamespace(
        json={
            "model_definitions": [
                {"model_definition_id": "md-allowed"},
                {"model_definition_id": "md-denied"},
            ]
        },
        data=None,
    )
    a.filter_list_gateway_model_definitions(resp)
    ids = [m["model_definition_id"] for m in json.loads(resp.data)["model_definitions"]]
    assert ids == ["md-allowed"]


def test_filter_list_gateway_secrets_drops_unreadable(monkeypatch):
    monkeypatch.setattr(a, "sender_is_admin", lambda: False)
    monkeypatch.setattr(a, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        a, "_role_based_read_predicate", lambda username, rt: lambda rid: rid == "sec-allowed"
    )
    resp = SimpleNamespace(
        json={"secrets": [{"secret_id": "sec-allowed"}, {"secret_id": "sec-denied"}]},
        data=None,
    )
    a.filter_list_gateway_secrets(resp)
    ids = [s["secret_id"] for s in json.loads(resp.data)["secrets"]]
    assert ids == ["sec-allowed"]


def test_secrets_config_redaction_registered():
    # The redaction filter must be wired for the secrets/config GET so _after_request runs it
    # (the path is otherwise excluded from the auto-built after-request handlers).
    key = (a.GATEWAY_SECRETS_CONFIG, "GET")
    assert a.AFTER_REQUEST_HANDLERS.get(key) is a.redact_secrets_config_for_non_admins


def test_secrets_config_redacts_passphrase_for_non_admin(monkeypatch):
    monkeypatch.setattr(a, "sender_is_admin", lambda: False)
    resp = SimpleNamespace(
        json={"secrets_available": True, "using_default_passphrase": True}, data=None
    )
    a.redact_secrets_config_for_non_admins(resp)
    body = json.loads(resp.data)
    assert body == {"secrets_available": True}
    assert "using_default_passphrase" not in body


def test_secrets_config_keeps_passphrase_for_admin(monkeypatch):
    monkeypatch.setattr(a, "sender_is_admin", lambda: True)
    resp = SimpleNamespace(
        json={"secrets_available": True, "using_default_passphrase": True}, data=None
    )
    a.redact_secrets_config_for_non_admins(resp)
    # Admin response is left untouched (data not rewritten).
    assert resp.data is None


def test_metric_history_bulk_interval_rest_prefix_is_gated():
    # The /api/2.0 twin was ungated when only the /ajax-api path had a validator.
    req = _Req("/api/2.0/mlflow/metrics/get-history-bulk-interval", "GET")
    assert a._find_validator(req) is not None


def test_demo_routes_gating():
    # generate is authenticated-open; delete hard-deletes the shared demo -> admin-only.
    gen = a._find_validator(_Req("/ajax-api/3.0/mlflow/demo/generate", "POST"))
    dele = a._find_validator(_Req("/ajax-api/3.0/mlflow/demo/delete", "POST"))
    assert gen is a._allow_authenticated
    assert dele is a.sender_is_admin
