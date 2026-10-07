import pytest
from flask import Flask
from werkzeug.datastructures import Authorization

from mlflow.server import auth, handlers
from mlflow.server.auth.db.models import SqlRolePermission
from mlflow.server.auth.permissions import NO_PERMISSIONS, READ
from mlflow.server.auth.routes import ADD_ROLE_PERMISSION, AJAX_ADD_ROLE_PERMISSION
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore as AuthStore
from mlflow.store.model_registry.sqlalchemy_workspace_store import (
    WorkspaceAwareSqlAlchemyStore as RegistryStore,
)
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.utils import workspace_context

from tests.store.tracking.sqlalchemy_store.conftest import create_test_span


@pytest.fixture
def large_scope_search_setup(tmp_path, db_uri, monkeypatch, request):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")
    auth_store = AuthStore()
    auth_store.init_db(f"sqlite:///{tmp_path / 'auth.db'}")
    request.addfinalizer(auth_store.engine.dispose)
    tracking_store = WorkspaceAwareSqlAlchemyStore(db_uri, (tmp_path / "artifacts").as_uri())
    request.addfinalizer(tracking_store._dispose_engine)
    registry_store = RegistryStore(db_uri)
    monkeypatch.setattr(auth, "store", auth_store)
    monkeypatch.setattr(auth, "_auth_initialized", True)
    monkeypatch.setattr(
        auth, "auth_config", auth.auth_config._replace(default_permission=NO_PERMISSIONS.name)
    )
    for module in (auth, handlers):
        monkeypatch.setattr(module, "_get_tracking_store", lambda: tracking_store)
        monkeypatch.setattr(module, "_get_model_registry_store", lambda: registry_store)
    username = "large-scope-reader"
    password = "supersecurepassword"
    user = auth_store.create_user(username, password)
    role = auth_store.create_role(name="large-scope", workspace="team-a")
    auth_store.assign_role_to_user(user.id, role.id)
    with workspace_context.WorkspaceContext("team-a"):
        yield auth_store, tracking_store, registry_store, role, username, password


@pytest.mark.parametrize("caller_filter", ["", "name = 'visible-a'"])
@pytest.mark.parametrize("scope_size", [800, 1100, 3400])
@pytest.mark.parametrize(
    ("collection", "handler", "resource_type", "response_key", "method"),
    [
        ("experiments", handlers._search_experiments, "experiment", "experiments", "GET"),
        ("experiments", handlers._search_experiments, "experiment", "experiments", "POST"),
        (
            "registered-models",
            handlers._search_registered_models,
            "registered_model",
            "registered_models",
            "GET",
        ),
        (
            "model-versions",
            handlers._search_model_versions,
            "registered_model",
            "model_versions",
            "GET",
        ),
    ],
)
def test_authenticated_search_with_large_scope(
    large_scope_search_setup,
    method,
    caller_filter,
    collection,
    handler,
    resource_type,
    response_key,
    scope_size,
    limit_sqlite_variables,
):
    auth_store, tracking, registry, role, username, password = large_scope_search_setup
    readable_ids = set()
    for name in ("hidden", "visible-a", "visible-b"):
        if resource_type == "experiment":
            resource_id = tracking.create_experiment(name)
        else:
            registry.create_registered_model(name)
            registry.create_model_version(name, source="/model", run_id=None)
            resource_id = name
        if name != "hidden":
            readable_ids.add(resource_id)
    with workspace_context.WorkspaceContext("team-b"):
        if resource_type == "experiment":
            readable_ids.add(tracking.create_experiment("outside-workspace"))
        else:
            registry.create_registered_model("outside-workspace")
            registry.create_model_version("outside-workspace", source="/model", run_id=None)
            readable_ids.add("outside-workspace")
    readable_ids.update(
        str(index + 10000) if resource_type == "experiment" else f"unused-{index}"
        for index in range(scope_size)
    )
    with auth_store.ManagedSessionMaker(read_only=False) as session:
        session.add_all([
            SqlRolePermission(
                role_id=role.id,
                resource_type=resource_type,
                resource_pattern=value,
                permission=READ.name,
            )
            for value in readable_ids
        ])

    limit_sqlite_variables(tracking.engine)
    if registry.engine is not tracking.engine:
        limit_sqlite_variables(registry.engine)
    if scope_size == 800:
        # Each list is below the old per-list threshold; together they exceed 999 bindings.
        key = "experiment_id" if resource_type == "experiment" else "name"
        caller_filter = (
            caller_filter + " AND " if caller_filter else ""
        ) + f"{key} IN ({', '.join(repr(value) for value in sorted(readable_ids))})"

    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    path = f"/api/2.0/mlflow/{collection}/search"
    app.add_url_rule(path, view_func=handler, methods=[method])
    headers = {
        "Authorization": Authorization(
            "basic", {"username": username, "password": password}
        ).to_header()
    }
    client = app.test_client()
    names = []
    params = {"filter": caller_filter, "max_results": 1, "order_by": ["name ASC"]}
    while True:
        kwargs = {"query_string": params} if method == "GET" else {"json": params}
        response = client.open(path, method=method, headers=headers, **kwargs)
        payload = response.json
        assert response.status_code == 200
        names.extend(item["name"] for item in payload.get(response_key, []))
        if not (token := payload.get("next_page_token")):
            break
        params["page_token"] = token
    assert names == (
        ["visible-a"] if "name = 'visible-a'" in caller_filter else ["visible-a", "visible-b"]
    )


@pytest.mark.parametrize("scope_size", [2, 3400])
@pytest.mark.parametrize("caller_filter", ["", "name LIKE '%model%'"])
@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
def test_authenticated_registered_model_search_refills_with_large_scope(
    large_scope_search_setup, limit_sqlite_variables, scope_size, caller_filter, prefix
):
    auth_store, _, registry, role, username, password = large_scope_search_setup
    hidden_names = ["a-hidden-model-0", "a-hidden-model-1"]
    visible_names = ["b-visible-model-0", "b-visible-model-1", "b-visible-model-2"]
    for name in ["0-ungranted-model", *hidden_names, *visible_names]:
        registry.create_registered_model(name)
    with workspace_context.WorkspaceContext("team-b"):
        registry.create_registered_model("outside-model")

    # Prompt grants on model names enter the union scope, then the response filter removes them.
    grants = [
        *(("prompt", name) for name in hidden_names),
        *(("registered_model", name) for name in visible_names),
        ("registered_model", "outside-model"),
        *(("registered_model", f"unused-{index}") for index in range(scope_size)),
    ]
    with auth_store.ManagedSessionMaker(read_only=False) as session:
        session.add_all([
            SqlRolePermission(
                role_id=role.id,
                resource_type=resource_type,
                resource_pattern=name,
                permission=READ.name,
            )
            for resource_type, name in grants
        ])
    limit_sqlite_variables(registry.engine)
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    path = f"{prefix}/2.0/mlflow/registered-models/search"
    app.add_url_rule(path, view_func=handlers._search_registered_models, methods=["GET"])
    headers = {
        "Authorization": Authorization(
            "basic", {"username": username, "password": password}
        ).to_header()
    }
    client = app.test_client()
    params = {"filter": caller_filter, "max_results": 2, "order_by": ["name ASC"]}
    response = client.get(path, query_string=params, headers=headers)
    assert response.status_code == 200
    assert [model["name"] for model in response.json["registered_models"]] == visible_names[:2]
    params["page_token"] = response.json["next_page_token"]
    response = client.get(path, query_string=params, headers=headers)
    assert response.status_code == 200
    assert [model["name"] for model in response.json["registered_models"]] == visible_names[2:]
    assert not response.json.get("next_page_token")


@pytest.mark.parametrize("has_valid_grant", [True, False])
@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize(
    ("endpoint", "method", "handler", "response_key"),
    [
        ("2.0/mlflow/experiments/search", "GET", handlers._search_experiments, "experiments"),
        ("2.0/mlflow/experiments/search", "POST", handlers._search_experiments, "experiments"),
        ("3.0/mlflow/traces/batchGet", "GET", handlers._batch_get_traces, "traces"),
        (
            "3.0/mlflow/traces/batchGetInfos",
            "POST",
            handlers._batch_get_trace_infos,
            "trace_infos",
        ),
        ("2.0/mlflow/logged-models/search", "POST", handlers._search_logged_models, "models"),
        ("3.0/mlflow/scorers/list", "GET", handlers._list_scorers, "scorers"),
    ],
)
def test_authenticated_queries_tolerate_legacy_nonnumeric_experiment_grants(
    large_scope_search_setup, has_valid_grant, prefix, endpoint, method, handler, response_key
):
    auth_store, tracking, _, role, username, password = large_scope_search_setup
    readable_experiment = tracking.create_experiment("readable")
    unreadable_experiment = tracking.create_experiment("unreadable")
    traces = ["tr-readable", "tr-unreadable", "tr-outside-workspace"]
    for experiment_id, trace_id in zip([readable_experiment, unreadable_experiment], traces[:2]):
        tracking.log_spans(experiment_id, [create_test_span(trace_id=trace_id)])
    with workspace_context.WorkspaceContext("team-b"):
        outside_experiment = tracking.create_experiment("outside-workspace")
        tracking.log_spans(outside_experiment, [create_test_span(trace_id=traces[2])])
    if response_key in {"models", "scorers"}:
        for experiment_id, workspace in [
            (readable_experiment, "team-a"),
            (unreadable_experiment, "team-a"),
            (outside_experiment, "team-b"),
        ]:
            with workspace_context.WorkspaceContext(workspace):
                if response_key == "models":
                    tracking.create_logged_model(experiment_id, f"model-{experiment_id}")
                else:
                    tracking.register_scorer(experiment_id, f"scorer-{experiment_id}", "{}")
    if has_valid_grant:
        for experiment_id in (readable_experiment, outside_experiment):
            auth_store.add_role_permission(role.id, "experiment", experiment_id, READ.name)
            if response_key == "scorers":
                auth_store.add_role_permission(
                    role.id,
                    "scorer",
                    auth_store._scorer_pattern(experiment_id, f"scorer-{experiment_id}"),
                    READ.name,
                )
    # Seed permissions accepted before validation was added to the role API.
    with auth_store.ManagedSessionMaker(read_only=False) as session:
        session.add_all([
            SqlRolePermission(
                role_id=role.id,
                resource_type="experiment",
                resource_pattern=pattern,
                permission=READ.name,
            )
            for pattern in ("obsolete-experiment", "", "1.5")
        ])

    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    path = f"{prefix}/{endpoint}"
    app.add_url_rule(path, view_func=handler, methods=[method])
    params = {"trace_ids": traces} if response_key in {"traces", "trace_infos"} else {}
    if response_key in {"experiments", "models"}:
        params["max_results"] = 10
    if response_key == "models":
        params["experiment_ids"] = [
            readable_experiment,
            unreadable_experiment,
            outside_experiment,
        ]
    kwargs = {"query_string": params} if method == "GET" else {"json": params}
    headers = {
        "Authorization": Authorization(
            "basic", {"username": username, "password": password}
        ).to_header()
    }
    response = app.test_client().open(path, method=method, headers=headers, **kwargs)

    assert response.status_code == 200
    results = response.json.get(response_key, [])
    if response_key in {"experiments", "models"}:
        infos = [model["info"] for model in results] if response_key == "models" else results
        assert [info["experiment_id"] for info in infos] == (
            [readable_experiment] if has_valid_grant else []
        )
    elif response_key == "scorers":
        assert [scorer["scorer_name"] for scorer in results] == (
            [f"scorer-{readable_experiment}"] if has_valid_grant else []
        )
    else:
        infos = [trace["trace_info"] for trace in results] if response_key == "traces" else results
        assert [info["trace_id"] for info in infos] == ([traces[0]] if has_valid_grant else [])


@pytest.mark.parametrize("resource_pattern", ["obsolete-experiment", "", "1.5", 1.5, None])
@pytest.mark.parametrize("path", [ADD_ROLE_PERMISSION, AJAX_ADD_ROLE_PERMISSION])
def test_role_permission_api_rejects_nonnumeric_experiment_patterns(
    large_scope_search_setup, resource_pattern, path
):
    auth_store, _, _, role, _, password = large_scope_search_setup
    auth_store.create_user("scope-admin", password, is_admin=True)
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    app.add_url_rule(path, view_func=auth.add_role_permission, methods=["POST"])
    response = app.test_client().post(
        path,
        json={
            "role_id": role.id,
            "resource_type": "experiment",
            "resource_pattern": resource_pattern,
            "permission": READ.name,
        },
        headers={
            "Authorization": Authorization(
                "basic", {"username": "scope-admin", "password": password}
            ).to_header()
        },
    )

    assert response.status_code == 400
    assert response.json["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "must be a valid integer" in response.json["message"]
    assert auth_store.list_role_permissions(role.id) == []


def test_experiment_scope_preserves_invalid_caller_filter_validation(large_scope_search_setup):
    auth_store, tracking, _, role, username, password = large_scope_search_setup
    experiment_id = tracking.create_experiment("readable")
    auth_store.add_role_permission(role.id, "experiment", experiment_id, READ.name)
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    path = "/api/2.0/mlflow/experiments/search"
    app.add_url_rule(path, view_func=handlers._search_experiments, methods=["POST"])
    response = app.test_client().post(
        path,
        json={"filter": "experiment_id IN ('obsolete-experiment')"},
        headers={
            "Authorization": Authorization(
                "basic", {"username": username, "password": password}
            ).to_header()
        },
    )

    assert response.status_code == 400
    assert response.json["error_code"] == "INVALID_PARAMETER_VALUE"
