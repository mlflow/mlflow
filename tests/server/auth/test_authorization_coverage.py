# Authorization-coverage guard for the basic-auth app.
#
# The basic-auth dispatcher authorizes via an allowlist and, historically, allowed
# any authenticated request whose route resolved no validator (fail-open). These
# tests turn "someone forgot to gate a new route" from a silent authorization gap
# into a failing test: every route registered in get_endpoints() must resolve to an
# explicit authorization decision, or be listed in the documented, temporarily-
# grandfathered debt list _KNOWN_UNGATED_ROUTE_MARKERS.

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


def test_metric_history_bulk_interval_rest_prefix_is_gated():
    # Regression: the interval bulk-metrics route is served under both /api/2.0 and
    # /ajax-api/2.0, but the validator was registered only for the ajax path, leaving
    # the /api/2.0 twin ungated.
    req = _Req("/api/2.0/mlflow/metrics/get-history-bulk-interval", "GET")
    assert a._find_validator(req) is not None


def test_demo_routes_gating():
    # generate is open to any authenticated user (shared onboarding data); delete
    # hard-deletes the shared demo experiment and is admin-only.
    gen = a._find_validator(_Req("/ajax-api/3.0/mlflow/demo/generate", "POST"))
    dele = a._find_validator(_Req("/ajax-api/3.0/mlflow/demo/delete", "POST"))
    assert gen is not None
    assert gen.__name__ == "_allow_authenticated"
    assert dele is not None
    assert dele.__name__ == "sender_is_admin"


def test_gateway_discovery_config_budget_gating():
    def vname(path, method="GET"):
        v = a._find_validator(_Req(path, method))
        return v.__name__ if v is not None else None

    # Discovery = authenticated-open (static capability lists, no tenant data).
    assert vname("/ajax-api/3.0/mlflow/gateway/supported-providers") == "_allow_authenticated"
    assert vname("/ajax-api/3.0/mlflow/gateway/supported-models") == "_allow_authenticated"
    # Config + budget reads = admin-only.
    assert vname("/ajax-api/3.0/mlflow/gateway/provider-config") == "sender_is_admin"
    assert vname("/ajax-api/3.0/mlflow/gateway/secrets/config") == "sender_is_admin"
    assert vname("/api/3.0/mlflow/gateway/budgets/get") == "sender_is_admin"
    assert vname("/api/3.0/mlflow/gateway/budgets/list") == "sender_is_admin"
    assert vname("/api/3.0/mlflow/gateway/budgets/windows") == "sender_is_admin"


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
    # D2: cross-resource gateway list endpoints are filtered after-request to rows
    # the caller can read.
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


def test_jobs_owner_check(monkeypatch):
    # D4-B: jobs get/cancel are gated on ownership — the recorded creator must match
    # the caller; a different creator or no creator is denied (admins bypass upstream).
    import mlflow.server.jobs as jobs_mod

    monkeypatch.setattr(a, "authenticate_request", lambda: SimpleNamespace(username="alice"))
    monkeypatch.setattr(a, "_get_request_param", lambda p: "job-1")

    monkeypatch.setattr(jobs_mod, "get_job", lambda jid: SimpleNamespace(creator="alice"))
    assert a.validate_is_job_owner() is True

    monkeypatch.setattr(jobs_mod, "get_job", lambda jid: SimpleNamespace(creator="bob"))
    assert a.validate_is_job_owner() is False

    monkeypatch.setattr(jobs_mod, "get_job", lambda jid: SimpleNamespace(creator=None))
    assert a.validate_is_job_owner() is False


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
