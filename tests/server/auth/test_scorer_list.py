import json
from types import SimpleNamespace
from unittest import mock

import pytest
from flask import Flask, g, request
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
def scorer_listing(request, tmp_path, db_uri, monkeypatch):
    enabled = request.param
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", str(enabled).lower())
    workspace = "scorer-team" if enabled else DEFAULT_WORKSPACE_NAME
    tracking_cls = WorkspaceAwareSqlAlchemyStore if enabled else SqlAlchemyStore
    tracking = tracking_cls(db_uri, str(tmp_path / "artifacts"))
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
    monkeypatch.setenv("MLFLOW_FLASK_SERVER_SECRET_KEY", "test-secret")
    monkeypatch.setenv("_MLFLOW_SGI_NAME", "flask")
    monkeypatch.setattr(auth, "_auth_initialized", False)
    with mock.patch.object(auth, "_init_store_for_app") as init_store:
        assert auth.create_app(app) is app
    init_store.assert_called_once_with()
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
            app=app,
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
    kwargs = {"query_string" if method == "GET" else "json": payload}
    response = env.client.open(
        f"/{prefix}/3.0/mlflow/scorers/list",
        method=method,
        headers={"X-Test-User": user},
        **kwargs,
    )
    assert response.status_code == 200, response.get_json()
    return response.get_json().get("scorers", [])


@pytest.mark.parametrize("prefix", ["api", "ajax-api"])
def test_scorer_only_grant_filters_payloads(scorer_listing, prefix):
    env = scorer_listing
    env.auth.grant_user_permission("reader", "scorer", f"{env.a}/toxicity", "READ")
    assert handlers.ListScorers not in auth.AFTER_REQUEST_PATH_HANDLERS
    with mock.patch.object(
        env.tracking,
        "_batch_resolve_endpoint_in_serialized_scorers",
        wraps=env.tracking._batch_resolve_endpoint_in_serialized_scorers,
    ) as resolve:
        for scope in [{}, {"experiment_id": env.a}, {"experiment_ids": [env.a, env.b]}]:
            scorers = _list(env, payload=scope, prefix=prefix)
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
    assert len(_list(env, payload={"experiment_ids": [env.a]})) == 3
    assert len(_list(env, payload={"experiment_ids": [env.b]})) == 1
    assert _list(env, payload={"experiment_ids": ["999999"]}) == []


def test_empty_grants_and_admin_requests_are_isolated(scorer_listing):
    env = scorer_listing
    assert _list(env) == []
    response = env.client.get("/api/3.0/mlflow/scorers/list", query_string={"experiment_id": env.a})
    assert response.status_code == 403
    assert len(_list(env, user="admin")) == 6
    assert len(_list(env, payload={"experiment_id": env.a}, user="admin")) == 3
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
    assert _list(env, payload={"experiment_ids": [env.a, foreign]}) == []


def test_auth_lookup_failure_does_not_drop_constraint(scorer_listing, monkeypatch):
    env = scorer_listing

    def fail(*args):
        raise RuntimeError("auth database unavailable")

    monkeypatch.setattr(env.auth, "list_role_grants_for_user_in_workspace", fail)
    with mock.patch.object(env.tracking, "list_scorers_across_experiments") as listing:
        response = env.client.get("/api/3.0/mlflow/scorers/list")
    assert response.status_code == 500
    listing.assert_not_called()


@pytest.mark.parametrize("resource_type", ["experiment", "scorer"])
def test_singular_scope_requires_a_grant_in_that_experiment(scorer_listing, resource_type):
    env = scorer_listing
    pattern = env.a if resource_type == "experiment" else f"{env.a}/absent"
    env.auth.grant_user_permission("reader", resource_type, pattern, "READ")
    assert len(_list(env, payload={"experiment_id": env.a})) == (
        3 if resource_type == "experiment" else 0
    )
    response = env.client.get("/api/3.0/mlflow/scorers/list", query_string={"experiment_id": env.b})
    assert response.status_code == 403
    assert _list(env, payload={"experiment_ids": [env.b]}) == []


@pytest.mark.parametrize("name", ["*", "a/b", "%2F", "毒性"])
def test_scorer_grant_names_are_literal(scorer_listing, name):
    env = scorer_listing
    env.tracking.register_scorer(env.a, name, "{}")
    env.tracking.register_scorer(env.a, "/", "{}")
    env.tracking.register_scorer(env.b, name, "{}")
    env.auth.grant_user_permission(
        "reader", "scorer", env.auth._scorer_pattern(env.a, name), "READ"
    )
    assert [(str(s["experiment_id"]), s["scorer_name"]) for s in _list(env)] == [(env.a, name)]


@pytest.mark.parametrize("missing_hook", [False, True])
def test_auth_initialization_requires_scorer_list_decision(
    scorer_listing, monkeypatch, missing_hook
):
    env = scorer_listing
    assert env.app.config["MLFLOW_REQUIRE_SCORER_LIST_AUTHORIZATION"] is True
    if missing_hook:
        monkeypatch.setitem(env.app.before_request_funcs, None, [])
    with (
        mock.patch.object(auth, "_find_validator", return_value=lambda: True) as validator,
        mock.patch.object(handlers, "_get_tracking_store") as tracking_store,
    ):
        response = env.client.get("/api/3.0/mlflow/scorers/list")
    assert response.status_code == 500
    assert response.json["error_code"] == "INTERNAL_ERROR"
    assert "authorization decision" in response.json["message"]
    if missing_hook:
        validator.assert_not_called()
    else:
        validator.assert_called_once()
    tracking_store.assert_not_called()


def test_admin_explicitly_authorizes_scorer_listing(scorer_listing):
    env = scorer_listing
    with env.app.test_request_context(headers={"X-Test-User": "admin"}):
        assert auth._before_request() is None
        assert "mlflow_scorer_filter" in g
        assert g.mlflow_scorer_filter is None


@pytest.mark.parametrize(
    ("state", "status", "code"),
    [
        ("malformed", 400, "INVALID_PARAMETER_VALUE"),
        ("missing", 404, "RESOURCE_DOES_NOT_EXIST"),
        ("deleted", 400, "INVALID_PARAMETER_VALUE"),
        ("foreign", 404, "RESOURCE_DOES_NOT_EXIST"),
    ],
)
@pytest.mark.parametrize("user", ["reader", "admin"])
def test_authorized_singular_scope_preserves_validation(scorer_listing, state, status, code, user):
    env = scorer_listing
    if state == "foreign" and not env.enabled:
        pytest.skip("Requires workspace isolation")
    if state == "malformed":
        eid = "invalid id"
        env.auth.grant_user_permission("reader", "experiment", "*", "READ")
    elif state == "missing":
        eid = "999999"
    elif state == "deleted":
        eid = env.a
        env.tracking.delete_experiment(eid)
    else:
        with WorkspaceContext("foreign-team"):
            eid = env.tracking.create_experiment("foreign-validation")
    if state != "malformed":
        env.auth.grant_user_permission("reader", "scorer", f"{eid}/absent", "READ")
    response = env.client.get(
        "/api/3.0/mlflow/scorers/list",
        query_string={"experiment_id": eid},
        headers={"X-Test-User": user},
    )
    assert response.status_code == status
    assert response.json["error_code"] == code


@pytest.mark.parametrize("user", ["reader", "admin"])
def test_plural_scope_drops_deleted_missing_and_foreign_experiments(scorer_listing, user):
    env = scorer_listing
    env.auth.grant_user_permission("reader", "experiment", env.a, "READ")
    env.auth.grant_user_permission("reader", "experiment", env.b, "READ")
    env.tracking.delete_experiment(env.b)
    ids = [env.a, env.b, "999999"]
    if env.enabled:
        with WorkspaceContext("foreign-team"):
            foreign = env.tracking.create_experiment("foreign-batch")
            env.tracking.register_scorer(foreign, "private", "{}")
        env.auth.grant_user_permission("reader", "experiment", foreign, "READ")
        ids.append(foreign)
    scorers = _list(env, payload={"experiment_ids": ids}, user=user)
    assert len(scorers) == 3
    assert {str(s["experiment_id"]) for s in scorers} == {env.a}


@pytest.mark.parametrize("alias", ["api", "ajax-api"])
def test_scorer_listing_auth_with_static_prefix(scorer_listing, monkeypatch, alias):
    env = scorer_listing
    monkeypatch.setenv(handlers.STATIC_PREFIX_ENV_VAR, "/custom-prefix")
    for path, handler, methods in handlers.get_endpoints():
        if handler is handlers._list_scorers:
            env.app.add_url_rule(path, view_func=handler, methods=methods)
    # These routes are generated at import time when the server starts with a prefix.
    for path, validator, methods in handlers.get_endpoints(auth.get_before_request_handler):
        if validator is auth.validate_can_read_scorer_list:
            for method in methods:
                monkeypatch.setitem(auth.BEFORE_REQUEST_VALIDATORS, (path, method), validator)
    env.auth.grant_user_permission("reader", "scorer", f"{env.a}/toxicity", "READ")
    scorers = _list(env, prefix=f"custom-prefix/{alias}")
    assert [(str(s["experiment_id"]), s["scorer_name"]) for s in scorers] == [(env.a, "toxicity")]


def test_workspace_admin_grant_lists_current_workspace(scorer_listing):
    env = scorer_listing
    env.auth.grant_user_permission("reader", "workspace", "*", "MANAGE")
    if env.enabled:
        with WorkspaceContext("foreign-team"):
            foreign = env.tracking.create_experiment("foreign-admin")
            env.tracking.register_scorer(foreign, "private", "{}")
    assert len(_list(env)) == 6


def test_scorer_listing_no_auth_app_is_independent(scorer_listing):
    env = scorer_listing
    no_auth = Flask("no-auth-scorers")
    no_auth.add_url_rule("/scorers", view_func=handlers._list_scorers, methods=["GET"])
    response = no_auth.test_client().get("/scorers")
    assert response.status_code == 200
    assert len(response.json["scorers"]) == 6
    assert _list(env) == []
