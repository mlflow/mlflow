# CI guard for the basic-auth dispatcher: every Flask route must resolve to an
# authorization decision, or be listed in the debt list _KNOWN_UNGATED_ROUTE_MARKERS.

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
