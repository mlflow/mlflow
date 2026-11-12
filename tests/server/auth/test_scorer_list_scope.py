import json
import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from flask import Flask, request
from sqlalchemy import event
from werkzeug.datastructures import Authorization

from mlflow.server import auth, handlers
from mlflow.server._request_keys import SCOPED_SCORER_LIST, SCORER_LIST_SCOPE_REQUIRED
from mlflow.server.auth.permissions import NO_PERMISSIONS, READ, USE, get_permission
from mlflow.server.auth.sqlalchemy_store import RoleGrantRow
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore as AuthStore
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.dbmodels.models import SqlScorer, SqlScorerVersion
from mlflow.store.tracking.scorer_list_scope import scorer_row_key
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME


def factory_gate(monkeypatch, grants, default="NO_PERMISSIONS", absent=None, workspace="ws"):
    monkeypatch.setattr(auth, "get_anchor_workspace", lambda *args: workspace)
    monkeypatch.setattr(auth, "_absent_permission", lambda *args: absent or get_permission(default))
    monkeypatch.setattr(auth, "auth_config", auth.auth_config._replace(default_permission=default))
    store = SimpleNamespace(
        get_user=lambda _: SimpleNamespace(id=1), list_grants=Mock(return_value=grants)
    )
    monkeypatch.setattr(auth, "store", store)
    gate = auth._scorer_listing_gate("user")
    assert store.list_grants.call_count == (workspace is not None)
    return gate


def assert_equivalent(gate, probes):
    scope = auth._compile_scorer_list_scope(gate)
    for e, n in probes:
        expected = (
            gate.retains("scorer_version")
            and gate.retains("experiment", e)
            and gate.retains("scorer", scorer_row_key(e, n))
        )
        assert (scope is None or scope.permits(e, n)) == expected, (e, n, scope)
    return scope


def test_generated_factory_equivalence(monkeypatch):
    rng = random.Random(97007)
    probes = [
        (e, n)
        for e in ("1", "2", "3", "opaque-id")
        for n in ("a", "a/b", "%2F", "*", "é", "unnamed")
    ]
    patterns = {
        "workspace": ["*"],
        "experiment": ["*", "1", "2", "01", "", "1.0", "opaque-id"],
        "scorer": ["*", "1/a", "1/a/b", "1/a%2Fb", "1/%2A", "2/%252F", "01/a", "opaque-id/a"],
        "scorer_version": ["*"],
    }
    for _ in range(400):
        grants = [
            RoleGrantRow(t, p, rng.choice(["READ", "DENY", "USE", "NO_PERMISSIONS"]))
            for t, ps in patterns.items()
            for p in ps
            for _ in range(rng.randrange(3))
        ]
        gate = factory_gate(
            monkeypatch,
            grants,
            rng.choice(["READ", "NO_PERMISSIONS"]),
            rng.choice([READ, USE, NO_PERMISSIONS]),
        )
        assert_equivalent(gate, probes)


@pytest.mark.parametrize("permission", ["READ", "DENY"])
@pytest.mark.parametrize("encoded", [False, True])
def test_dead_raw_slash_does_not_become_live(monkeypatch, permission, encoded):
    grants = [
        RoleGrantRow("experiment", "1", "READ"),
        RoleGrantRow("scorer", "1/a%2Fb" if encoded else "1/a/b", permission),
    ]
    gate = factory_gate(monkeypatch, grants, "READ" if permission == "DENY" else "NO_PERMISSIONS")
    scope = assert_equivalent(gate, [("1", "a/b"), ("1", "other")])
    assert (scope is None or scope.permits("1", "a/b")) == (
        encoded if permission == "READ" else not encoded
    )


def test_factory_admin_and_unresolved(monkeypatch):
    grants = [
        RoleGrantRow("workspace", "*", "MANAGE"),
        RoleGrantRow("experiment", "1", "DENY"),
        RoleGrantRow("scorer", "1/a", "DENY"),
    ]
    assert assert_equivalent(factory_gate(monkeypatch, grants), [("1", "a"), ("2", "a")]) is None
    scope = assert_equivalent(factory_gate(monkeypatch, grants, workspace=None), [("1", "a")])
    assert scope is not None
    assert not scope.permits("1", "a")


def test_acceptance_and_preserved_workspace_use(monkeypatch):
    grants = [RoleGrantRow("experiment", "1", "READ"), RoleGrantRow("scorer", "1/a", "READ")]
    scope = assert_equivalent(
        factory_gate(monkeypatch, grants), [("1", "a"), ("1", "b"), ("2", "a")]
    )
    assert scope.permits("1", "a")
    assert not scope.permits("1", "b")
    # Shipped READ default doesn't close siblings just because one grant exists.
    assert assert_equivalent(factory_gate(monkeypatch, grants, "READ"), [("1", "b")]) is None
    gate = factory_gate(monkeypatch, [RoleGrantRow("experiment", "1", "READ")])
    assert not assert_equivalent(gate, [("1", "a")]).permits("1", "a")
    gate = factory_gate(
        monkeypatch,
        [RoleGrantRow("workspace", "*", "USE"), RoleGrantRow("experiment", "1", "READ")],
    )
    assert assert_equivalent(gate, [("1", "a")]) is None


def test_auth_failure_propagates(monkeypatch):
    monkeypatch.setattr(
        auth, "get_anchor_workspace", Mock(side_effect=RuntimeError("lookup failed"))
    )
    with pytest.raises(RuntimeError, match="lookup failed"):
        auth._scorer_listing_gate("user")


@pytest.fixture(params=[False, True], ids=["workspace-disabled", "workspace-enabled"])
def listing_workspace(request, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", str(request.param).lower())
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        yield request.param


@pytest.fixture
def scoped_auth_app(tmp_path, monkeypatch, listing_workspace, db_uri):
    monkeypatch.setenv("MLFLOW_FLASK_SERVER_SECRET_KEY", "isolated-test-secret")
    monkeypatch.setenv("MLFLOW_BASIC_AUTH_FAIL_CLOSED", "true")
    monkeypatch.setattr(
        auth,
        "auth_config",
        auth.auth_config._replace(
            default_permission="NO_PERMISSIONS",
            authorization_function="custom_identity:authenticate",
        ),
    )
    auth_store = AuthStore()
    auth_store.init_db(f"sqlite:///{tmp_path / 'auth.db'}")
    for user in ("alice", "bob", "admin"):
        auth_store.create_user(user, "test-password", is_admin=user == "admin")
    monkeypatch.setattr(auth, "store", auth_store)
    monkeypatch.setattr(auth, "_init_store_for_app", lambda: None)

    def identity():
        return Authorization("basic", {"username": request.headers.get("X-Test-User", "alice")})

    monkeypatch.setattr(auth, "get_auth_func", lambda configured: identity)
    store_type = WorkspaceAwareSqlAlchemyStore if listing_workspace else SqlAlchemyStore
    tracking = store_type(db_uri, str(tmp_path / "artifacts"))
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: tracking)
    monkeypatch.setattr(auth, "_get_tracking_store", lambda: tracking)
    e = tracking.create_experiment("parent")
    for name in ("allowed", "denied", "a/b", "A", "a "):
        tracking.register_scorer(e, name, json.dumps({"name": name}))
    app = auth.create_app(Flask(__name__))
    assert app.config[SCORER_LIST_SCOPE_REQUIRED] is True
    for index, path in enumerate((
        "/api/3.0/mlflow/scorers/list",
        "/ajax-api/3.0/mlflow/scorers/list",
    )):
        app.add_url_rule(path, f"list{index}", handlers._list_scorers)
    yield app, tracking, auth_store, e
    tracking.engine.dispose()
    auth_store.engine.dispose()


def grant(auth_store, user, grants):
    role = auth_store.create_role(f"role-{user}", DEFAULT_WORKSPACE_NAME)
    auth_store.assign_role_to_user(auth_store.get_user(user).id, role.id)
    for t, p, permission in grants:
        auth_store.add_role_permission(role.id, t, p, permission)


@pytest.mark.parametrize("veto", ["scorer", "scorer_version"])
@pytest.mark.parametrize("has_versions", [False, True])
@pytest.mark.parametrize("legacy_lookup", [False, True])
def test_online_configs_deleted_experiment_retains_veto(
    scoped_auth_app, monkeypatch, veto, has_versions, legacy_lookup
):
    app, tracking, auth_store, e = scoped_auth_app
    scorer = tracking.get_scorer(e, "a/b")
    tracking.upsert_online_scoring_config(e, "a/b", 0.0)
    if not has_versions:
        tracking.delete_scorer(e, "a/b", version=1)
    if legacy_lookup:
        # Exercise the optional default and an old-signature backend's historical lookup.
        monkeypatch.setattr(
            tracking,
            "_get_scorer_names_by_ids",
            lambda ids: AbstractStore._get_scorer_names_by_ids(tracking, ids),
        )
        monkeypatch.setattr(
            tracking,
            "list_scorers_across_experiments",
            lambda ids: [scorer] if has_versions else [],
        )
    else:
        monkeypatch.setattr(
            tracking,
            "list_scorers_across_experiments",
            Mock(side_effect=AssertionError("payload listing must not resolve auth names")),
        )
    grant(
        auth_store,
        "alice",
        [
            ("experiment", e, "READ"),
            (veto, scorer_row_key(e, "a/b") if veto == "scorer" else "*", "DENY"),
        ],
    )
    grant(auth_store, "bob", [("experiment", e, "MANAGE")])
    auth_store.create_user("other", "test-password")
    paths = [f"/{prefix}/3.0/mlflow/scorers/online-configs" for prefix in ("api", "ajax-api")]
    for index, path in enumerate(paths):
        app.add_url_rule(path, f"configs{index}", handlers._get_online_scoring_configs)

    for deleted in (False, True):
        if deleted:
            tracking.delete_experiment(e)
        for path in paths:
            for user, expected in (
                ("alice", 403 if has_versions else 200),
                ("bob", 200),
                ("admin", 200),
                ("other", 403),
            ):
                response = app.test_client().get(
                    path,
                    query_string={"scorer_ids": scorer.scorer_id},
                    headers={"X-Test-User": user},
                )
                assert response.status_code == expected, response.json
                if expected == 200:
                    assert response.json["configs"][0]["scorer_id"] == scorer.scorer_id


@pytest.mark.parametrize("selector", ["singular", "plural", "omitted"])
@pytest.mark.parametrize(
    "policy", ["exact", "broad_deny", "version_veto", "coarse_exact", "coarse_deny"]
)
def test_full_auth_scope_denied_versions_and_resolver(
    scoped_auth_app, monkeypatch, selector, policy
):
    app, tracking, auth_store, e = scoped_auth_app
    grants = [("experiment", e, "READ")]
    if policy in ("exact", "coarse_exact"):
        grants.append(("scorer", scorer_row_key(e, "allowed"), "READ"))
        expected = {"allowed"}
    else:
        grants.extend([("scorer", "*", "READ"), ("scorer", scorer_row_key(e, "denied"), "DENY")])
        expected = {"allowed", "a/b", "A", "a "}
        if policy == "version_veto":
            grants.append(("scorer_version", "*", "DENY"))
            expected = set()
    grant(auth_store, "alice", grants)
    oracle = auth._scorer_listing_gate("alice")
    assert {
        name
        for name in ("allowed", "denied", "a/b", "A", "a ")
        if oracle.retains("scorer_version")
        and oracle.retains("experiment", e)
        and oracle.retains("scorer", scorer_row_key(e, name))
    } == expected
    if policy.startswith("coarse"):
        original = tracking._scorer_identity_page
        monkeypatch.setattr(
            tracking,
            "_scorer_identity_page",
            lambda session, selector, after: original(session, True, after),
        )
    with tracking.ManagedSessionMaker(read_only=False) as session:
        denied = (
            session
            .query(SqlScorerVersion)
            .join(SqlScorer)
            .filter(SqlScorer.experiment_id == int(e), SqlScorer.scorer_name == "denied")
            .one()
        )
        denied.serialized_scorer = "invalid JSON: must never load/resolve"
    loaded = []

    def load(row, context):
        loaded.append(json.loads(row.serialized_scorer)["name"])

    resolver = Mock(side_effect=lambda values: values)
    monkeypatch.setattr(tracking, "_batch_resolve_endpoint_in_serialized_scorers", resolver)
    event.listen(SqlScorerVersion, "load", load)
    query = (
        {"experiment_id": e}
        if selector == "singular"
        else {"experiment_ids": [e]}
        if selector == "plural"
        else {}
    )
    try:
        response = app.test_client().get("/api/3.0/mlflow/scorers/list", query_string=query)
    finally:
        event.remove(SqlScorerVersion, "load", load)
    assert response.status_code == 200, response.json
    assert {s["scorer_name"] for s in response.json.get("scorers", [])} == expected
    assert set(loaded) == expected
    if expected:
        assert {json.loads(s)["name"] for s in resolver.call_args.args[0]} == expected
    else:
        resolver.assert_not_called()


def test_custom_identity_admin_aliases_and_request_isolation(scoped_auth_app, monkeypatch):
    app, tracking, auth_store, e = scoped_auth_app
    grant(
        auth_store,
        "alice",
        [("experiment", e, "READ"), ("scorer", scorer_row_key(e, "allowed"), "READ")],
    )
    decisions = []

    @app.after_request
    def observe(response):
        decisions.append(getattr(request, SCOPED_SCORER_LIST, "missing"))
        return response

    with app.app_context():
        for path in ("/api/3.0/mlflow/scorers/list", "/ajax-api/3.0/mlflow/scorers/list"):
            for user in ("alice", "admin", "bob"):
                response = app.test_client().get(path, headers={"X-Test-User": user})
                assert response.status_code == 200, response.json
                names = {s["scorer_name"] for s in response.json.get("scorers", [])}
                assert names == (
                    {"allowed"}
                    if user == "alice"
                    else {"allowed", "denied", "a/b", "A", "a "}
                    if user == "admin"
                    else set()
                )
                if user == "admin":
                    assert decisions[-1] is None
                else:
                    assert decisions[-1] != "missing"
                    assert decisions[-1] is not None
    # Missing scoper cannot succeed unrestricted, including fail-closed route recognition.
    scopes = dict(auth.REQUEST_SCOPE_HANDLERS)
    scopes.pop(("/api/3.0/mlflow/scorers/list", "GET"))
    monkeypatch.setattr(auth, "REQUEST_SCOPE_HANDLERS", scopes)
    response = app.test_client().get("/api/3.0/mlflow/scorers/list")
    assert response.status_code == 500
    assert response.json["error_code"] == "INTERNAL_ERROR"


@pytest.mark.parametrize("prefix", ["", "/tracking"])
def test_head_scope_dispatch_selection_parent_gate_and_missing_writer(
    scoped_auth_app, monkeypatch, prefix
):
    app, tracking, auth_store, e = scoped_auth_app
    grant(
        auth_store,
        "alice",
        [("experiment", e, "READ"), ("scorer", scorer_row_key(e, "allowed"), "READ")],
    )
    foreign = tracking.create_experiment("unreadable-parent")
    tracking.register_scorer(foreign, "other", "{}")
    monkeypatch.setenv(auth.STATIC_PREFIX_ENV_VAR, prefix)
    scopes = {
        (path, method): scoper
        for path, scoper, methods in auth.get_endpoints(auth.REQUEST_SCOPE_FUNCTIONS.get)
        for method in methods
        if scoper is auth._scope_list_scorers_query
    }
    validators = {
        (path, method): validator
        for path, validator, methods in auth.get_endpoints(auth.get_before_request_handler)
        for method in methods
        if validator is auth.validate_can_read_scorer_list
    }
    monkeypatch.setattr(auth, "REQUEST_SCOPE_HANDLERS", scopes)
    monkeypatch.setattr(auth, "BEFORE_REQUEST_VALIDATORS", validators)
    paths = [f"{prefix}/{alias}/3.0/mlflow/scorers/list" for alias in ("api", "ajax-api")]
    if prefix:
        for index, path in enumerate(paths):
            app.add_url_rule(path, f"prefixed{index}", handlers._list_scorers)
    decisions = []

    @app.after_request
    def observe(response):
        decisions.append(getattr(request, SCOPED_SCORER_LIST, "missing"))
        return response

    singular = Mock(wraps=tracking.list_scorers)
    batch = Mock(wraps=tracking.list_scorers_across_experiments)
    monkeypatch.setattr(tracking, "list_scorers", singular)
    monkeypatch.setattr(tracking, "list_scorers_across_experiments", batch)
    client = app.test_client()
    for fail_closed in (False, True):
        monkeypatch.setenv("MLFLOW_BASIC_AUTH_FAIL_CLOSED", str(fail_closed).lower())
        for path in paths:
            for user in ("alice", "admin"):
                headers = {"X-Test-User": user}
                for query in ({"experiment_id": e}, {"experiment_ids": [e]}, {}):
                    singular.reset_mock()
                    batch.reset_mock()
                    response = client.head(path, query_string=query, headers=headers)
                    expected = 403 if fail_closed and user == "alice" else 200
                    assert response.status_code == expected
                    assert response.data == b""
                    if expected == 200:
                        if user == "admin":
                            assert decisions[-1] is None
                        else:
                            assert decisions[-1].permits(e, "allowed")
                            assert not decisions[-1].permits(e, "denied")
                        if "experiment_id" in query:
                            assert singular.call_args.args == (e,)
                        elif "experiment_ids" in query:
                            assert not singular.called
                            assert batch.call_args.args == ([e],)
                    else:
                        singular.assert_not_called()
                        batch.assert_not_called()
                # Accepted HEAD must not lose the singular parent permission check.
                response = client.head(
                    path, query_string={"experiment_id": foreign}, headers=headers
                )
                assert response.status_code == (200 if user == "admin" else 403)
                assert response.data == b""
                # Query selection wins over the JSON body, just as for GET.
                response = client.head(
                    path,
                    query_string={"experiment_id": foreign},
                    json={"experiment_id": e},
                    headers=headers,
                )
                assert response.status_code == (200 if user == "admin" else 403)
                assert response.data == b""
                if not fail_closed or user == "admin":
                    response = client.head(
                        path,
                        query_string={"experiment_id": e, "experiment_ids": [e]},
                        headers=headers,
                    )
                    assert response.status_code == 400
                    assert response.data == b""
                    for selection in ({"experimentId": e}, {"experiment_ids": []}):
                        singular.reset_mock()
                        batch.reset_mock()
                        response = client.head(path, json=selection, headers=headers)
                        assert response.status_code == 200
                        assert response.data == b""
                        if "experimentId" in selection:
                            assert singular.call_args.args == (e,)
                        else:
                            singular.assert_not_called()
                            assert batch.call_args.args == ([],)
            # Removing the writer must still fail closed, never imply unrestricted.
            key = (path, "GET")
            writer = scopes.pop(key)
            try:
                for user in ("alice", "admin"):
                    response = client.head(path, headers={"X-Test-User": user})
                    assert response.status_code == (403 if fail_closed and user == "alice" else 500)
                    assert response.data == b""
            finally:
                scopes[key] = writer


@pytest.mark.parametrize(
    "selection",
    [
        {"experiment_id": "E"},
        {"experimentId": "E"},
        {"experiment_ids": ["E", "E"]},
        {"experimentIds": ["E"]},
        {"experiment_ids": []},
        {"experiment_id": ""},
    ],
)
def test_actual_parser_get_json_selection(scoped_auth_app, selection):
    app, tracking, auth_store, e = scoped_auth_app
    grant(
        auth_store,
        "alice",
        [("experiment", e, "READ"), ("scorer", scorer_row_key(e, "allowed"), "READ")],
    )
    selection = {
        k: [e if v == "E" else v for v in value]
        if isinstance(value, list)
        else e
        if value == "E"
        else value
        for k, value in selection.items()
    }
    response = app.test_client().get("/api/3.0/mlflow/scorers/list", json=selection)
    assert response.status_code == 200, response.json
    assert [s["scorer_name"] for s in response.json.get("scorers", [])] == (
        [] if selection.get("experiment_ids") == [] else ["allowed"]
    )


@pytest.mark.parametrize(
    "selection",
    [
        {"experiment_id": "1", "experimentId": "2"},
        {"experiment_id": "1", "experiment_ids": []},
        {"experiment_ids": ["invalid id"]},
        {"experiment_ids": [3]},
        {"experiment_id": 3},
    ],
)
def test_original_validation_before_empty_scope(scoped_auth_app, selection):
    app, tracking, auth_store, e = scoped_auth_app
    grant(auth_store, "alice", [("experiment", "*", "READ"), ("scorer_version", "*", "DENY")])
    response = app.test_client().get("/api/3.0/mlflow/scorers/list", json=selection)
    assert response.status_code == 400, response.json
    assert response.json["error_code"] == "INVALID_PARAMETER_VALUE"


def test_grant_sql_hygiene_before_open_ended_metadata(scoped_auth_app, monkeypatch):
    app, tracking, auth_store, e = scoped_auth_app
    grant(
        auth_store,
        "alice",
        [
            ("experiment", e, "READ"),
            ("scorer", "*", "READ"),
        ],
    )
    original = auth_store.list_grants
    monkeypatch.setattr(
        auth_store,
        "list_grants",
        lambda *args: (
            original(*args)
            + [
                RoleGrantRow("experiment", "opaque-id", "READ"),
                RoleGrantRow("experiment", "9" * 64, "READ"),
            ]
        ),
    )
    validate = Mock(wraps=tracking.filter_active_experiment_ids)
    monkeypatch.setattr(tracking, "filter_active_experiment_ids", validate)
    response = app.test_client().get("/api/3.0/mlflow/scorers/list")
    assert response.status_code == 200, response.json
    validate.assert_called_once_with([e])


def test_query_precedence_and_empty_scope_singular_resource_errors(scoped_auth_app):
    app, tracking, auth_store, e = scoped_auth_app
    grant(auth_store, "alice", [("experiment", "*", "READ"), ("scorer_version", "*", "DENY")])
    client = app.test_client()
    response = client.get(
        "/api/3.0/mlflow/scorers/list",
        query_string={"experiment_id": e},
        json={"experiment_id": "999999"},
    )
    assert response.status_code == 200
    assert client.get("/api/3.0/mlflow/scorers/list?experiment_id=999999").status_code == (
        403 if tracking.supports_workspaces else 404
    )
    tracking.delete_experiment(e)
    assert client.get(f"/api/3.0/mlflow/scorers/list?experiment_id={e}").status_code == 400


def test_scope_isolation_across_apps_and_workspace_changes(scoped_auth_app):
    app, tracking, auth_store, e = scoped_auth_app
    grant(
        auth_store,
        "alice",
        [("experiment", e, "READ"), ("scorer", scorer_row_key(e, "allowed"), "READ")],
    )
    plain = Flask("nonparticipating")
    plain.add_url_rule("/list", view_func=handlers._list_scorers)
    assert not plain.config.get(SCORER_LIST_SCOPE_REQUIRED)
    with app.app_context():
        response = app.test_client().get("/api/3.0/mlflow/scorers/list")
        assert [s["scorer_name"] for s in response.json["scorers"]] == ["allowed"]
        response = plain.test_client().get(f"/list?experiment_id={e}")
        assert len(response.json["scorers"]) == 5
        if tracking.supports_workspaces:
            with WorkspaceContext("another-workspace"):
                foreign = tracking.create_experiment("parent")
                tracking.register_scorer(foreign, "allowed", "{}")
                response = app.test_client().get("/api/3.0/mlflow/scorers/list")
                assert response.status_code == 200
                assert not response.json.get("scorers")
        response = app.test_client().get("/api/3.0/mlflow/scorers/list")
        assert [s["scorer_name"] for s in response.json["scorers"]] == ["allowed"]
        auth_store.create_scorer_permission(e, "allowed", "alice", "DENY")
        response = app.test_client().get("/api/3.0/mlflow/scorers/list")
        assert response.status_code == 200
        assert not response.json.get("scorers")
