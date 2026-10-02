import json
from types import SimpleNamespace
from unittest import mock

import pytest
from flask import Flask, request
from werkzeug.datastructures import Authorization

from mlflow.server import auth, handlers
from mlflow.server.auth.config import read_auth_config
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore as AuthStore
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

pytestmark = pytest.mark.notrackingurimock


@pytest.fixture(params=[False, True], ids=["workspace-disabled", "workspace-enabled"])
def scorer_listing(request, tmp_path, monkeypatch):
    enabled = request.param
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", str(enabled).lower())
    workspace = "scorer-team" if enabled else DEFAULT_WORKSPACE_NAME
    tracking_cls = WorkspaceAwareSqlAlchemyStore if enabled else SqlAlchemyStore
    tracking = tracking_cls(f"sqlite:///{tmp_path}/tracking.db", str(tmp_path / "artifacts"))
    auth_store = AuthStore()
    auth_store.init_db(f"sqlite:///{tmp_path}/auth.db")
    auth_store.create_user("reader", "reader-password")
    auth_store.create_user("admin", "admin-password", is_admin=True)
    monkeypatch.setattr(auth, "store", auth_store)
    monkeypatch.setattr(
        auth,
        "auth_config",
        read_auth_config()._replace(default_permission="NO_PERMISSIONS"),
    )
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: tracking)
    monkeypatch.setattr(auth, "authenticate_request", _authenticate_test_request)
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    for path, handler, methods in handlers.get_endpoints():
        if handler is handlers._list_scorers:
            app.add_url_rule(path, view_func=handler, methods=methods)
    with WorkspaceContext(workspace):
        exp_a = tracking.create_experiment("scorer-a")
        exp_b = tracking.create_experiment("scorer-b")
        for eid in [exp_a, exp_b]:
            for name in ["toxicity", "other", "*/%2F/'毒性"]:
                tracking.register_scorer(eid, name, json.dumps({"key": f"{eid}/{name}"}))
        tracking.register_scorer(exp_a, "toxicity", '{"latest": true}')
        yield SimpleNamespace(
            client=app.test_client(),
            auth=auth_store,
            tracking=tracking,
            a=exp_a,
            b=exp_b,
            workspace=workspace,
            enabled=enabled,
        )
    auth_store.engine.dispose()
    tracking._dispose_engine()


def _authenticate_test_request():
    return Authorization("basic", {"username": request.headers.get("X-Test-User", "reader")})


def _list(env, method="GET", payload=None, prefix="api", user="reader"):
    payload = dict(payload or {})
    if method == "GET" and "scorer_filter" in payload:
        payload["scorer_filter"] = json.dumps(payload["scorer_filter"])
    kwargs = {"query_string" if method == "GET" else "json": payload}
    response = env.client.open(
        f"/{prefix}/3.0/mlflow/scorers/list",
        method=method,
        headers={"X-Test-User": user},
        **kwargs,
    )
    assert response.status_code == 200, response.get_json()
    return response.get_json().get("scorers", [])


@pytest.mark.parametrize(("method", "prefix"), [("GET", "api"), ("POST", "ajax-api")])
def test_scorer_only_grant_filters_payloads(scorer_listing, method, prefix):
    env = scorer_listing
    env.auth.grant_user_permission("reader", "scorer", f"{env.a}/toxicity", "READ")
    assert handlers.ListScorers not in auth.AFTER_REQUEST_PATH_HANDLERS
    with mock.patch.object(
        env.tracking,
        "_batch_resolve_endpoint_in_serialized_scorers",
        wraps=env.tracking._batch_resolve_endpoint_in_serialized_scorers,
    ) as resolve:
        for scope in [{}, {"experiment_id": env.a}, {"experiment_ids": [env.a, env.b]}]:
            scorers = _list(env, method, scope, prefix)
            assert [
                (str(s["experiment_id"]), s["scorer_name"], s["scorer_version"]) for s in scorers
            ] == [(env.a, "toxicity", 2)]
            assert resolve.call_args.args[0] == ['{"latest": true}']


def test_mixed_experiment_and_scorer_grants(scorer_listing):
    env = scorer_listing
    env.auth.grant_user_permission("reader", "experiment", env.a, "READ")
    encoded_name = env.auth._scorer_pattern(env.b, "*/%2F/'毒性")
    env.auth.grant_user_permission("reader", "scorer", encoded_name, "READ")
    expected = {(env.a, name) for name in ["toxicity", "other", "*/%2F/'毒性"]}
    expected.add((env.b, "*/%2F/'毒性"))
    assert {(str(s["experiment_id"]), s["scorer_name"]) for s in _list(env)} == expected
    # Caller selection is intersected with grants, including inside one experiment.
    assert (
        _list(
            env,
            "POST",
            {"scorer_filter": {"scorers": [{"experiment_id": env.b, "scorer_name": "other"}]}},
        )
        == []
    )
    assert len(_list(env, "POST", {"scorer_filter": {"experiment_ids": [env.a]}})) == 3
    assert _list(env, "POST", {"experiment_ids": []}) == []
    assert _list(env, "POST", {"scorer_filter": {}}) == []


def test_empty_grants_and_admin_requests_are_isolated(scorer_listing):
    env = scorer_listing
    assert _list(env) == []
    assert _list(env, payload={"experiment_id": env.a}) == []
    assert len(_list(env, user="admin")) == 6
    assert _list(env, "POST", {"scorer_filter": {}}, user="admin") == []
    assert _list(env) == []


@pytest.mark.parametrize("resource_type", ["experiment", "scorer", "workspace"])
def test_list_scorers_wildcard_grants(scorer_listing, resource_type):
    env = scorer_listing
    env.auth.grant_user_permission(
        "reader", resource_type, "*", "MANAGE" if resource_type == "workspace" else "READ"
    )
    assert len(_list(env)) == 6


def test_list_scorers_default_permission(scorer_listing, monkeypatch):
    env = scorer_listing
    monkeypatch.setattr(auth, "auth_config", auth.auth_config._replace(default_permission="READ"))
    assert len(_list(env)) == (0 if env.enabled else 6)
    if env.enabled:
        monkeypatch.setattr(auth, "_get_workspace_store", lambda: None)
        monkeypatch.setattr(
            auth,
            "get_default_workspace_optional",
            lambda _: (SimpleNamespace(name=env.workspace), True),
        )
        monkeypatch.setattr(
            auth, "auth_config", auth.auth_config._replace(grant_default_workspace_access=True)
        )
        assert len(_list(env)) == 6


def test_foreign_workspace_grants_do_not_authorize(scorer_listing):
    env = scorer_listing
    if not env.enabled:
        pytest.skip("Requires workspace isolation")
    with WorkspaceContext("foreign-team"):
        env.auth.grant_user_permission("reader", "experiment", env.a, "READ")
        env.auth.grant_user_permission("reader", "scorer", f"{env.a}/toxicity", "READ")
        foreign = env.tracking.create_experiment("foreign-scorers")
        env.tracking.register_scorer(foreign, "toxicity", '{"private": true}')
    env.auth.grant_user_permission("reader", "scorer", f"{foreign}/toxicity", "READ")
    assert _list(env) == []
    assert _list(env, "POST", {"experiment_ids": [env.a, foreign]}) == []


def test_auth_lookup_failure_does_not_drop_constraint(scorer_listing, monkeypatch):
    env = scorer_listing

    def fail(*args):
        raise RuntimeError("auth database unavailable")

    monkeypatch.setattr(env.auth, "list_role_grants_for_user_in_workspace", fail)
    with mock.patch.object(env.tracking, "list_scorers_across_experiments") as listing:
        response = env.client.get("/api/3.0/mlflow/scorers/list")
    assert response.status_code == 500
    listing.assert_not_called()
