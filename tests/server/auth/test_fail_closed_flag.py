# Tests for the MLFLOW_BASIC_AUTH_FAIL_CLOSED feature flag and the fail-closed
# decision predicates.
#
# Note: the end-to-end denial path is intentionally NOT reachable via a real route
# today. Every currently-ungated route is grandfathered in
# _KNOWN_UNGATED_ROUTE_MARKERS, so with the flag on they still proceed; the flag
# only starts denying once a marker is removed (as its validator lands). These
# tests therefore cover the flag wiring and the decision predicates directly.

from mlflow.environment_variables import MLFLOW_BASIC_AUTH_FAIL_CLOSED
from mlflow.server import auth as a


class _Req:
    def __init__(self, path, method="GET"):
        self.path = path
        self.method = method


def test_flag_defaults_off():
    assert MLFLOW_BASIC_AUTH_FAIL_CLOSED.get() is False


def test_flag_reads_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_BASIC_AUTH_FAIL_CLOSED", "true")
    assert MLFLOW_BASIC_AUTH_FAIL_CLOSED.get() is True


def test_debt_list_is_empty():
    # Every known ungated route family has been gated with a real validator, so the
    # grandfather list is empty and the fail-closed flag can be flipped on safely.
    assert a._KNOWN_UNGATED_ROUTE_MARKERS == ()


def test_unlisted_route_is_denied_when_fail_closed():
    # A brand-new route with no validator, not grandfathered, and not authorized
    # elsewhere is exactly what the fail-closed branch denies when the flag is on.
    req = _Req("/api/3.0/mlflow/brand-new-feature/do-thing", "POST")
    assert not a._is_known_ungated_route(req.path)
    assert not a._authorized_outside_before_request(req)


def test_authorized_elsewhere_is_recognized():
    # Public and handler-internal-authz routes must never be denied even with the
    # flag on.
    assert a._authorized_outside_before_request(_Req("/api/3.0/mlflow/server-info"))
    assert a._authorized_outside_before_request(_Req("/api/2.0/mlflow/runs/search", "POST"))
