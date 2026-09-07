# Tests for the MLFLOW_BASIC_AUTH_FAIL_CLOSED flag: wiring, the decision predicates,
# and the end-to-end 403 that _before_request returns when the flag is on.

from mlflow.environment_variables import MLFLOW_BASIC_AUTH_FAIL_CLOSED
from mlflow.server import auth as a


class _Req:
    def __init__(self, path, method="GET"):
        self.path = path
        self.method = method


def test_flag_defaults_on():
    assert MLFLOW_BASIC_AUTH_FAIL_CLOSED.get() is True


def test_flag_reads_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_BASIC_AUTH_FAIL_CLOSED", "true")
    assert MLFLOW_BASIC_AUTH_FAIL_CLOSED.get() is True
    # The opt-out direction matters now that the default is True.
    monkeypatch.setenv("MLFLOW_BASIC_AUTH_FAIL_CLOSED", "false")
    assert MLFLOW_BASIC_AUTH_FAIL_CLOSED.get() is False


def test_debt_list_is_empty():
    # All families gated -> the flag can be flipped on safely.
    assert a._KNOWN_UNGATED_ROUTE_MARKERS == ()


def test_unlisted_route_is_denied_when_fail_closed():
    req = _Req("/api/3.0/mlflow/brand-new-feature/do-thing", "POST")
    assert not a._is_known_ungated_route(req.path)
    assert not a._authorized_outside_before_request(req)


def test_authorized_elsewhere_is_recognized():
    assert a._authorized_outside_before_request(_Req("/api/3.0/mlflow/server-info"))
    assert a._authorized_outside_before_request(_Req("/api/2.0/mlflow/runs/search", "POST"))


def test_public_suffix_not_matched_as_incidental_tail():
    # Anchored to an API-version prefix, so an unrelated route ending in the same
    # segments is not treated as authorized.
    assert not a._authorized_outside_before_request(
        _Req("/api/2.0/mlflow/evil/passthrough/mlflow/runs/search", "POST")
    )
    assert a._authorized_outside_before_request(_Req("/ajax-api/2.0/mlflow/runs/search", "POST"))


def test_public_routes_recognized_under_static_prefix(monkeypatch):
    # Under --static-prefix the request path carries the prefix; public/internal routes
    # must still be recognized (else they wrongly fail closed).
    from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR

    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/custom-prefix")
    assert a._authorized_outside_before_request(_Req("/custom-prefix/api/3.0/mlflow/server-info"))
    assert a._authorized_outside_before_request(
        _Req("/custom-prefix/api/2.0/mlflow/runs/search", "POST")
    )
    assert a._authorized_outside_before_request(_Req("/custom-prefix/graphql"))


def test_web_ui_entry_points_resolve_a_validator():
    # `/` serves the SPA shell, `/version` the version string the bundle reads, and
    # ui-telemetry the UI's own config. None carries tenant data, and none is a handler
    # endpoint, so get_endpoints() never surfaced them to the coverage guard -- without a
    # validator the fail-closed net denies them and a non-admin browser gets a plain
    # "Permission denied" instead of MLflow.
    for path, method in (
        ("/", "GET"),
        ("/version", "GET"),
        ("/ajax-api/3.0/mlflow/ui-telemetry", "GET"),
        ("/ajax-api/3.0/mlflow/ui-telemetry", "POST"),
    ):
        validator = a._find_validator(_Req(path, method))
        assert validator is not None, f"{method} {path} resolves no authorization decision"
        assert validator() is True, f"{method} {path} is denied for an authenticated user"


def test_web_ui_entry_points_resolve_a_validator_under_static_prefix(monkeypatch):
    # The keys are built at import time from the static prefix, so this reloads the
    # route table with one configured rather than reusing the module-level map.
    import importlib

    from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR

    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/custom-prefix")
    routes = importlib.reload(importlib.import_module("mlflow.server.auth.routes"))
    try:
        assert routes.UI_TELEMETRY == "/custom-prefix/ajax-api/3.0/mlflow/ui-telemetry"
        # HOME and SERVER_VERSION stay bare; the auth module prefixes them itself.
        assert routes.HOME == "/"
        assert routes.SERVER_VERSION == "/version"
    finally:
        monkeypatch.delenv(STATIC_PREFIX_ENV_VAR)
        importlib.reload(routes)


def _fake_basic_auth():
    from werkzeug.datastructures import Authorization

    return Authorization("basic", {"username": "u"})


def test_before_request_returns_403_for_ungated_route_when_fail_closed(monkeypatch):
    import flask

    monkeypatch.setenv("MLFLOW_BASIC_AUTH_FAIL_CLOSED", "true")
    monkeypatch.setattr(a, "authenticate_request", _fake_basic_auth)
    monkeypatch.setattr(a, "sender_is_admin", lambda: False)

    app = flask.Flask(__name__)
    with app.test_request_context("/api/3.0/mlflow/brand-new-feature/do-thing", method="POST"):
        resp = a._before_request()
    assert resp is not None
    assert resp.status_code == 403


def test_before_request_allows_ungated_route_when_flag_off(monkeypatch):
    import flask

    monkeypatch.setenv("MLFLOW_BASIC_AUTH_FAIL_CLOSED", "false")
    monkeypatch.setattr(a, "authenticate_request", _fake_basic_auth)
    monkeypatch.setattr(a, "sender_is_admin", lambda: False)

    app = flask.Flask(__name__)
    with app.test_request_context("/api/3.0/mlflow/brand-new-feature/do-thing", method="POST"):
        resp = a._before_request()
    assert resp is None


def test_before_request_serves_the_web_ui_to_a_non_admin_when_fail_closed(monkeypatch):
    # Regression: with the net on, a non-admin got 403 on `GET /` and the browser showed
    # the string "Permission denied" instead of the UI. Admins were unaffected because
    # sender_is_admin() returns before the authorization branch, so this must pin the
    # non-admin case specifically.
    import flask

    monkeypatch.setenv("MLFLOW_BASIC_AUTH_FAIL_CLOSED", "true")
    monkeypatch.setattr(a, "authenticate_request", _fake_basic_auth)
    monkeypatch.setattr(a, "sender_is_admin", lambda: False)

    app = flask.Flask(__name__)
    for path, method in (
        ("/", "GET"),
        ("/version", "GET"),
        ("/ajax-api/3.0/mlflow/ui-telemetry", "GET"),
        ("/ajax-api/3.0/mlflow/ui-telemetry", "POST"),
    ):
        with app.test_request_context(path, method=method):
            resp = a._before_request()
        assert resp is None, f"{method} {path} denied for a non-admin: {resp}"
