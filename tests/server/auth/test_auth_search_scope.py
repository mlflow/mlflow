import base64
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from cachetools import TTLCache
from flask import Flask
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from starlette.testclient import TestClient
from werkzeug.datastructures import Authorization

from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException, RestException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, TEMPORARILY_UNAVAILABLE
from mlflow.protos.model_registry_pb2 import SearchModelVersions
from mlflow.protos.service_pb2 import SearchLoggedModels
from mlflow.server import auth, handlers
from mlflow.server.auth.db.models import SqlRolePermission
from mlflow.server.auth.permissions import DENY, MANAGE, NO_PERMISSIONS, READ, USE
from mlflow.server.auth.routes import ADD_ROLE_PERMISSION, AJAX_ADD_ROLE_PERMISSION
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore as AuthStore
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.rest_store import RestStore
from mlflow.store.model_registry.sqlalchemy_workspace_store import (
    WorkspaceAwareSqlAlchemyStore as RegistryStore,
)
from mlflow.store.tracking.databricks_rest_store import DatabricksTracingRestStore
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.rest_store import RestStore as TrackingRestStore
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.utils import workspace_context
from mlflow.utils.proto_json_utils import parse_dict
from mlflow.utils.rest_utils import MlflowHostCreds, verify_rest_response
from mlflow.utils.search_utils import SearchExperimentsUtils
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.server.test_mcp_server_api import _create_registry_fastapi_app
from tests.store.tracking.sqlalchemy_store.conftest import create_test_span


@pytest.fixture
def large_scope_search_setup(tmp_path, db_uri, monkeypatch, request):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")
    monkeypatch.setattr(
        auth,
        "_RESOURCE_WORKSPACE_CACHE",
        TTLCache(
            maxsize=auth._RESOURCE_WORKSPACE_CACHE.maxsize,
            ttl=auth._RESOURCE_WORKSPACE_CACHE.ttl,
        ),
    )
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
def test_authenticated_registry_search_with_large_scope(
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
        # Combined with the authorization scope values, this pushes the total bindings past
        # SQLite's 999-variable limit.
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


@pytest.mark.parametrize("experiment_ids", [None, ["invalid-id"]])
def test_logged_model_scope_preserves_caller_validation(large_scope_search_setup, experiment_ids):
    auth_store, tracking, _, role, username, password = large_scope_search_setup
    readable = tracking.create_experiment("readable")
    auth_store.add_role_permission(role.id, "experiment", readable, READ.name)
    app = Flask(__name__)
    app.before_request(auth._before_request)
    path = "/api/2.0/mlflow/logged-models/search"
    app.add_url_rule(path, view_func=handlers._search_logged_models, methods=["POST"])
    payload = {} if experiment_ids is None else {"experiment_ids": experiment_ids}
    response = app.test_client().post(
        path,
        json=payload,
        headers={
            "Authorization": Authorization(
                "basic", {"username": username, "password": password}
            ).to_header()
        },
    )
    assert response.status_code == 400
    assert response.json["error_code"] == "INVALID_PARAMETER_VALUE"


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


@pytest.mark.parametrize("field_name", ["experiment_ids", "experimentIds"])
@pytest.mark.parametrize("requested_scope", ["omitted", "readable", "unreadable", "empty"])
@pytest.mark.parametrize("grant", ["named", "wildcard", "wildcard-with-deny", "trace-denied"])
@pytest.mark.parametrize("infos", [False, True])
def test_batch_trace_response_enforces_scope_when_backend_ignores_it(
    large_scope_search_setup, monkeypatch, field_name, requested_scope, grant, infos
):
    auth_store, tracking, _, role, username, password = large_scope_search_setup
    readable = tracking.create_experiment("readable")
    unreadable = tracking.create_experiment("unreadable")
    for experiment_id, trace_id in [(readable, "tr-readable"), (unreadable, "tr-unreadable")]:
        tracking.log_spans(experiment_id, [create_test_span(trace_id=trace_id)])
    auth_store.add_role_permission(role.id, "workspace", "*", USE.name)
    auth_store.add_role_permission(
        role.id, "experiment", "*" if grant.startswith("wildcard") else readable, READ.name
    )
    if grant == "wildcard-with-deny":
        auth_store.add_role_permission(role.id, "experiment", unreadable, DENY.name)
    if grant == "trace-denied":
        auth_store.add_role_permission(role.id, "trace", "*", DENY.name)
    method_name = "batch_get_trace_infos" if infos else "batch_get_traces"
    original = getattr(tracking, method_name)
    batch_get = Mock(
        side_effect=lambda trace_ids, location=None, experiment_ids=None: original(
            trace_ids, location
        )
    )
    monkeypatch.setattr(tracking, method_name, batch_get)
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    path = "/api/3.0/mlflow/traces/" + ("batchGetInfos" if infos else "batchGet")
    app.add_url_rule(
        path,
        view_func=handlers._batch_get_trace_infos if infos else handlers._batch_get_traces,
        methods=["POST" if infos else "GET"],
    )
    payload = {"trace_ids": ["tr-readable", "tr-unreadable"]}
    if requested_scope != "omitted":
        payload[field_name] = {"readable": [readable], "unreadable": [unreadable], "empty": []}[
            requested_scope
        ]
    kwargs = {"json": payload} if infos else {"query_string": payload}
    response = app.test_client().open(
        path,
        method="POST" if infos else "GET",
        headers={
            "Authorization": Authorization(
                "basic", {"username": username, "password": password}
            ).to_header()
        },
        **kwargs,
    )
    if grant == "trace-denied":
        assert response.status_code == 403
        batch_get.assert_not_called()
        return
    assert response.status_code == 200
    batch_get.assert_called_once()
    rows = response.json.get("trace_infos" if infos else "traces", [])
    ids = [row["trace_id"] if infos else row["trace_info"]["trace_id"] for row in rows]
    # An empty list is omitted by GET query encoding.
    unrestricted = requested_scope == "omitted" or (not infos and requested_scope == "empty")
    expected = []
    if unrestricted or requested_scope == "readable":
        expected.append("tr-readable")
    if grant == "wildcard" and (unrestricted or requested_scope == "unreadable"):
        expected.append("tr-unreadable")
    assert sorted(ids) == sorted(expected)


@pytest.mark.parametrize("explicit_scope", [False, True])
def test_batch_trace_implicit_scope_preserves_databricks_compatibility(
    large_scope_search_setup, monkeypatch, explicit_scope
):
    auth_store, _, _, role, username, password = large_scope_search_setup
    auth_store.add_role_permission(role.id, "experiment", "1", READ.name)
    tracking = DatabricksTracingRestStore(lambda: MlflowHostCreds("https://example.com"))
    resolve_warehouse = Mock(return_value=None)
    call_endpoint = Mock(return_value=SimpleNamespace(traces=[]))
    monkeypatch.setattr(tracking, "_resolve_sql_warehouse_id", resolve_warehouse)
    monkeypatch.setattr(tracking, "_call_endpoint", call_endpoint)
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: tracking)
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    path = "/api/3.0/mlflow/traces/batchGet"
    app.add_url_rule(path, view_func=handlers._batch_get_traces, methods=["GET"])
    payload = {"trace_ids": ["tr-test"]}
    if explicit_scope:
        payload["experiment_ids"] = ["1"]
    response = app.test_client().get(
        path,
        query_string=payload,
        headers={
            "Authorization": Authorization(
                "basic", {"username": username, "password": password}
            ).to_header()
        },
    )
    assert response.status_code == (400 if explicit_scope else 200)
    if explicit_scope:
        assert "experiment_ids" in response.json["message"]
        resolve_warehouse.assert_not_called()
        call_endpoint.assert_not_called()
    else:
        resolve_warehouse.assert_called_once_with()
        call_endpoint.assert_called_once()


@pytest.mark.parametrize("readable_names", [["a1", "a3"], ["a1", "a3", "a5"]])
@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("opaque_tokens", [False, True])
def test_model_version_pagination_preserves_readable_results(
    large_scope_search_setup, monkeypatch, readable_names, prefix, opaque_tokens
):
    auth_store, _, registry, role, username, password = large_scope_search_setup
    for name in ["a1", "a2", "a3", "a4", "a5", "a6"]:
        registry.create_registered_model(name)
        registry.create_model_version(name, source="/model", run_id=None)
    for name in readable_names:
        auth_store.add_role_permission(role.id, "registered_model", name, READ.name)
    if opaque_tokens:
        search = registry.search_model_versions
        tokens = {}

        def search_with_opaque_tokens(**kwargs):
            if token := kwargs.get("page_token"):
                kwargs["page_token"] = tokens[token]
            page = search(**kwargs)
            if page.token:
                opaque = f"opaque-version-cursor-{len(tokens)}"
                tokens[opaque] = page.token
                page.token = opaque
            return page

        monkeypatch.setattr(registry, "search_model_versions", search_with_opaque_tokens)
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    path = f"{prefix}/2.0/mlflow/model-versions/search"
    app.add_url_rule(path, view_func=handlers._search_model_versions, methods=["GET"])
    params = {"max_results": 2, "order_by": ["name ASC"]}
    pages = []
    client = app.test_client()
    for _ in range(4):
        response = client.get(
            path,
            query_string=params,
            headers={
                "Authorization": Authorization(
                    "basic", {"username": username, "password": password}
                ).to_header()
            },
        )
        assert response.status_code == 200
        pages.append([version["name"] for version in response.json.get("model_versions", [])])
        if not (token := response.json.get("next_page_token")):
            break
        params["page_token"] = token
    else:
        pytest.fail("Pagination did not terminate")
    # A full page retains its backend token even when only unreadable rows remain.
    assert pages == [readable_names[:2], readable_names[2:]]


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("empty_pages", [1, 3])
@pytest.mark.parametrize("readable_final_page", [True, False])
def test_model_version_refill_follows_empty_remote_pages(
    large_scope_search_setup, monkeypatch, prefix, empty_pages, readable_final_page
):
    auth_store, _, _, role, username, password = large_scope_search_setup
    auth_store.add_role_permission(role.id, "registered_model", "visible", READ.name)
    remote = RestStore(lambda: MlflowHostCreds("https://example.com"))
    final_page = SearchModelVersions.Response()
    if readable_final_page:
        final_page.model_versions.add(name="visible", version="1", creation_timestamp=1)
    tokens = [f"opaque-cursor-{index}" for index in range(empty_pages + 1)]
    rpc = Mock(
        side_effect=[
            *(SearchModelVersions.Response(next_page_token=token) for token in tokens),
            final_page,
        ]
    )
    monkeypatch.setattr(remote, "_call_endpoint", rpc)
    for module in (auth, handlers):
        monkeypatch.setattr(module, "_get_model_registry_store", lambda: remote)
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    path = f"{prefix}/2.0/mlflow/model-versions/search"
    app.add_url_rule(path, view_func=handlers._search_model_versions, methods=["GET"])
    response = app.test_client().get(
        path,
        query_string={"max_results": 1, "filter": "name != 'excluded'", "order_by": "name ASC"},
        headers={
            "Authorization": Authorization(
                "basic", {"username": username, "password": password}
            ).to_header()
        },
    )
    assert response.status_code == 200
    assert [version["name"] for version in response.json.get("model_versions", [])] == (
        ["visible"] if readable_final_page else []
    )
    assert not response.json.get("next_page_token")
    assert rpc.call_count == len(tokens) + 1
    requests = [json.loads(call.args[1]) for call in rpc.call_args_list]
    assert [request.get("page_token") for request in requests] == [None, *tokens]
    for call, request in zip(rpc.call_args_list, requests):
        assert call.args[0] == SearchModelVersions
        assert request["filter"] == "name != 'excluded'"
        assert request["order_by"] == ["name ASC"]
        assert request["max_results"] == 1


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("outcome", ["created", "duplicate", "denied"])
def test_mcp_server_creation_reports_granted_actions(
    large_scope_search_setup, monkeypatch, prefix, outcome
):
    auth_store, tracking, _, role, username, _ = large_scope_search_setup
    auth_store.add_role_permission(role.id, "workspace", "*", USE.name)
    name = "com.example/new-server"
    if outcome == "duplicate":
        tracking.create_mcp_server(name=name)
    elif outcome == "denied":
        auth_store.add_role_permission(role.id, "mcp_server", name, DENY.name)
    user = auth_store.get_user(username)
    monkeypatch.setattr(auth, "_authenticate_fastapi_request", lambda _request: user)
    monkeypatch.setattr(
        auth,
        "resolve_workspace_for_request_if_enabled",
        lambda *_args: SimpleNamespace(name="team-a"),
    )
    app = _create_registry_fastapi_app()

    @app.middleware("http")
    async def workspace_middleware(request, call_next):
        with workspace_context.WorkspaceContext("team-a"):
            return await call_next(request)

    auth.add_fastapi_permission_middleware(app)
    client = TestClient(app)
    path = f"{prefix}/3.0/mlflow/mcp-servers"
    created = client.post(path, json={"name": name})
    if outcome == "created":
        assert created.status_code == 200
        fetched = client.get(f"{path}/{name}")
        assert fetched.status_code == 200
        assert (
            created.json()["allowed_actions"]
            == fetched.json()["allowed_actions"]
            == ["USE", "UPDATE", "DELETE", "MANAGE"]
        )
        grants = auth_store.list_mcp_server_permissions(username)
        assert [(grant.name, grant.permission) for grant in grants] == [(name, "MANAGE")]
    else:
        assert created.status_code == (400 if outcome == "duplicate" else 403)
        assert auth_store.list_mcp_server_permissions(username) == []
        assert client.get(f"{path}/{name}").status_code == 403


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("id_format", ["leading_zero", "plus", "whitespace"])
@pytest.mark.parametrize("grant", ["named", "wildcard", "wildcard-with-deny"])
@pytest.mark.parametrize(
    ("collection", "field_name"),
    [
        ("logged-models", "experiment_ids"),
        ("traces", "experiment_ids"),
        ("traces", "experimentIds"),
        ("trace-infos", "experiment_ids"),
        ("trace-infos", "experimentIds"),
    ],
)
def test_experiment_scope_accepts_numeric_id_forms(
    large_scope_search_setup, prefix, field_name, id_format, grant, collection
):
    auth_store, tracking, _, role, username, password = large_scope_search_setup
    experiment_id = tracking.create_experiment("visible")
    auth_store.add_role_permission(
        role.id, "experiment", experiment_id if grant == "named" else "*", READ.name
    )
    if grant == "wildcard-with-deny":
        auth_store.add_role_permission(role.id, "experiment", experiment_id, DENY.name)
    caller_id = {
        "leading_zero": f"0{experiment_id}",
        "plus": f"+{experiment_id}",
        "whitespace": f" {experiment_id} ",
    }[id_format]
    payload = {field_name: [caller_id]}
    if collection == "logged-models":
        model = tracking.create_logged_model(experiment_id, name="visible")
        path = f"{prefix}/2.0/mlflow/logged-models/search"
        handler = handlers._search_logged_models
        method = "POST"
        response_key = "models"
        expected_id = model.model_id
    else:
        tracking.log_spans(experiment_id, [create_test_span(trace_id="tr-visible")])
        payload["trace_ids"] = ["tr-visible"]
        infos = collection == "trace-infos"
        path = f"{prefix}/3.0/mlflow/traces/" + ("batchGetInfos" if infos else "batchGet")
        handler = handlers._batch_get_trace_infos if infos else handlers._batch_get_traces
        method = "POST" if infos else "GET"
        response_key = "trace_infos" if infos else "traces"
        expected_id = "tr-visible"
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    app.add_url_rule(path, view_func=handler, methods=[method])
    kwargs = {"query_string": payload} if method == "GET" else {"json": payload}
    response = app.test_client().open(
        path,
        method=method,
        headers={
            "Authorization": Authorization(
                "basic", {"username": username, "password": password}
            ).to_header()
        },
        **kwargs,
    )
    assert response.status_code == 200
    rows = response.json.get(response_key, [])
    if collection == "logged-models":
        returned_ids = [row["info"]["model_id"] for row in rows]
    else:
        infos = [row["trace_info"] for row in rows] if collection == "traces" else rows
        returned_ids = [info["trace_id"] for info in infos]
    assert returned_ids == ([] if grant == "wildcard-with-deny" else [expected_id])


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("empty_pages", [1, 3])
@pytest.mark.parametrize("final_page_kind", ["readable", "empty"])
def test_logged_model_search_preserves_empty_remote_pages(
    large_scope_search_setup,
    logged_model_search_client,
    monkeypatch,
    prefix,
    empty_pages,
    final_page_kind,
):
    auth_store, _, _, role, _, _ = large_scope_search_setup
    auth_store.add_role_permission(role.id, "experiment", "1", READ.name)
    remote = TrackingRestStore(lambda: MlflowHostCreds("https://example.com"))
    final_page = SearchLoggedModels.Response()
    if final_page_kind == "readable":
        info = final_page.models.add().info
        info.model_id = "m-visible"
        info.experiment_id = "1"
    tokens = [f"opaque-logged-model-cursor-{index}" for index in range(empty_pages)]
    rpc = Mock(
        side_effect=[
            *(SearchLoggedModels.Response(next_page_token=token) for token in tokens),
            final_page,
        ]
    )
    monkeypatch.setattr(remote, "_call_endpoint", rpc)
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: remote)
    payload = {
        "experiment_ids": ["1"],
        "max_results": 1,
        "filter": "name != 'excluded'",
        "order_by": [{"field_name": "name", "ascending": True}],
        "datasets": [{"dataset_name": "eval", "dataset_digest": "digest"}],
    }
    client, headers = logged_model_search_client
    for index in range(len(tokens) + 1):
        response = client.post(
            f"{prefix}/2.0/mlflow/logged-models/search", json=payload, headers=headers
        )
        assert response.status_code == 200
        assert rpc.call_count == index + 1
        assert [model["info"]["model_id"] for model in response.json.get("models", [])] == (
            ["m-visible"] if index == len(tokens) and final_page_kind == "readable" else []
        )
        assert (response.json.get("next_page_token") or None) == (
            tokens[index] if index < len(tokens) else None
        )
        if index < len(tokens):
            payload["page_token"] = tokens[index]
    requests = [json.loads(call.args[1]) for call in rpc.call_args_list]
    assert [query.get("page_token") for query in requests] == [None, *tokens]
    for call, query in zip(rpc.call_args_list, requests):
        assert call.args[0] == SearchLoggedModels
        for field in ("experiment_ids", "max_results", "filter", "order_by", "datasets"):
            assert query[field] == payload[field]


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("id_format", ["leading_zero", "plus", "whitespace"])
@pytest.mark.parametrize("grant", ["named", "wildcard"])
def test_logged_model_numeric_id_forms_preserve_pagination(
    large_scope_search_setup, prefix, id_format, grant
):
    auth_store, tracking, _, role, username, password = large_scope_search_setup
    experiment_id = tracking.create_experiment("visible")
    models = [tracking.create_logged_model(experiment_id, name=name) for name in ("a", "b")]
    auth_store.add_role_permission(
        role.id, "experiment", experiment_id if grant == "named" else "*", READ.name
    )
    caller_id = {
        "leading_zero": f"0{experiment_id}",
        "plus": f"+{experiment_id}",
        "whitespace": f" {experiment_id} ",
    }[id_format]
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    path = f"{prefix}/2.0/mlflow/logged-models/search"
    app.add_url_rule(path, view_func=handlers._search_logged_models, methods=["POST"])
    client = app.test_client()
    payload = {
        "experiment_ids": [caller_id],
        "max_results": 1,
        "order_by": [{"field_name": "name", "ascending": True}],
    }
    returned_ids = []
    for _ in range(3):
        response = client.post(
            path,
            json=payload,
            headers={
                "Authorization": Authorization(
                    "basic", {"username": username, "password": password}
                ).to_header()
            },
        )
        assert response.status_code == 200
        returned_ids.extend(model["info"]["model_id"] for model in response.json.get("models", []))
        if not (token := response.json.get("next_page_token")):
            break
        payload["page_token"] = token
    else:
        pytest.fail("Pagination did not terminate")
    assert returned_ids == [model.model_id for model in models]


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
def test_logged_model_json_scope_preserves_required_field_spelling(
    large_scope_search_setup, prefix
):
    auth_store, tracking, _, role, username, password = large_scope_search_setup
    experiment_id = tracking.create_experiment("visible")
    auth_store.add_role_permission(role.id, "experiment", experiment_id, READ.name)
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    path = f"{prefix}/2.0/mlflow/logged-models/search"
    app.add_url_rule(path, view_func=handlers._search_logged_models, methods=["POST"])
    response = app.test_client().post(
        path,
        json={"experimentIds": [experiment_id]},
        headers={
            "Authorization": Authorization(
                "basic", {"username": username, "password": password}
            ).to_header()
        },
    )
    assert response.status_code == 400
    assert response.json["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "experiment_ids" in response.json["message"]


@pytest.mark.parametrize("backend", ["file", "sql"])
@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("grant", ["named", "wildcard", "wildcard-with-deny"])
@pytest.mark.parametrize("max_results", [1, 2])
@pytest.mark.parametrize("id_format", ["plus", "leading_zero", "whitespace"])
def test_logged_model_scope_deduplicates_numeric_id_forms(
    large_scope_search_setup, tmp_path, monkeypatch, backend, prefix, grant, max_results, id_format
):
    auth_store, tracking, _, role, username, password = large_scope_search_setup
    if backend == "file":
        monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "false")
        monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
        tracking = FileStore(str(tmp_path / "file-store"))
        role = auth_store.create_role(name="file-reader", workspace=DEFAULT_WORKSPACE_NAME)
        auth_store.assign_role_to_user(auth_store.get_user(username).id, role.id)
        for module in (auth, handlers):
            monkeypatch.setattr(module, "_get_tracking_store", lambda: tracking)
    first = tracking.create_experiment("first")
    second = tracking.create_experiment("second")
    excluded = tracking.create_experiment("excluded")
    first_model = tracking.create_logged_model(first, name="a")
    second_model = tracking.create_logged_model(second, name="b")
    tracking.create_logged_model(excluded, name="c")
    if grant == "named":
        for experiment_id in (first, second):
            auth_store.add_role_permission(role.id, "experiment", experiment_id, READ.name)
    else:
        auth_store.add_role_permission(role.id, "experiment", "*", READ.name)
        if grant == "wildcard-with-deny":
            auth_store.add_role_permission(role.id, "experiment", second, DENY.name)

    def alias(experiment_id):
        return {
            "plus": f"+{experiment_id}",
            "leading_zero": f"0{experiment_id}",
            "whitespace": f" {experiment_id} ",
        }[id_format]

    search = Mock(wraps=tracking.search_logged_models)
    monkeypatch.setattr(tracking, "search_logged_models", search)
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    path = f"{prefix}/2.0/mlflow/logged-models/search"
    app.add_url_rule(path, view_func=handlers._search_logged_models, methods=["POST"])
    payload = {
        "experiment_ids": [second, alias(second), first, alias(first)],
        "max_results": max_results,
        "order_by": [{"field_name": "name", "ascending": True}],
    }
    returned_ids = []
    client = app.test_client()
    for _ in range(4):
        previous_calls = search.call_count
        response = client.post(
            path,
            json=payload,
            headers={
                "Authorization": Authorization(
                    "basic", {"username": username, "password": password}
                ).to_header()
            },
        )
        assert response.status_code == 200
        assert search.call_count == previous_calls + 1
        returned_ids.extend(model["info"]["model_id"] for model in response.json.get("models", []))
        if not (token := response.json.get("next_page_token")):
            break
        payload["page_token"] = token
    else:
        pytest.fail("Pagination did not terminate")
    assert returned_ids == (
        [first_model.model_id]
        if grant == "wildcard-with-deny"
        else [first_model.model_id, second_model.model_id]
    )
    search.assert_called()
    for call in search.call_args_list:
        assert call.kwargs["experiment_ids"] == (
            [first] if grant == "wildcard-with-deny" else [second, first]
        )


@pytest.fixture
def logged_model_search_client(large_scope_search_setup):
    _, _, _, _, username, password = large_scope_search_setup
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    for prefix in ("/api", "/ajax-api"):
        app.add_url_rule(
            f"{prefix}/2.0/mlflow/logged-models/search",
            endpoint=prefix,
            view_func=handlers._search_logged_models,
            methods=["POST"],
        )
    headers = {
        "Authorization": Authorization(
            "basic", {"username": username, "password": password}
        ).to_header()
    }
    return app.test_client(), headers


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("grant", ["named", "wildcard", "wildcard-with-deny"])
@pytest.mark.parametrize("id_format", ["canonical", "plus", "leading_zero", "whitespace"])
@pytest.mark.parametrize("token_ids", ["canonical", "raw"])
def test_logged_model_legacy_duplicate_id_tokens_fail_without_retry(
    large_scope_search_setup,
    logged_model_search_client,
    monkeypatch,
    prefix,
    grant,
    id_format,
    token_ids,
):
    auth_store, tracking, _, role, _, _ = large_scope_search_setup
    first = tracking.create_experiment("first")
    second = tracking.create_experiment("second")
    hidden = tracking.create_experiment("hidden")
    models = [
        tracking.create_logged_model(experiment_id, name=name)
        for experiment_id, name in [(first, "a"), (second, "b"), (first, "c"), (second, "d")]
    ]
    tracking.create_logged_model(hidden, name="0-hidden")
    if grant == "named":
        for experiment_id in (first, second):
            auth_store.add_role_permission(role.id, "experiment", experiment_id, READ.name)
    else:
        auth_store.add_role_permission(role.id, "experiment", "*", READ.name)
        if grant == "wildcard-with-deny":
            auth_store.add_role_permission(role.id, "experiment", hidden, DENY.name)

    def alias(experiment_id):
        return {
            "canonical": experiment_id,
            "plus": f"+{experiment_id}",
            "leading_zero": f"0{experiment_id}",
            "whitespace": f" {experiment_id} ",
        }[id_format]

    requested_ids = [second, alias(second), first, alias(first), second]
    legacy_ids = requested_ids if token_ids == "raw" else [second, second, first, first, second]
    options = {
        "filter_string": "name != 'excluded'",
        "max_results": 1,
        "order_by": [
            {
                "field_name": "name",
                "ascending": True,
                "dataset_name": None,
                "dataset_digest": None,
            }
        ],
    }
    old_page = tracking.search_logged_models(experiment_ids=legacy_ids, **options)
    assert [model.model_id for model in old_page] == [models[0].model_id]
    assert old_page.token
    search = Mock(wraps=tracking.search_logged_models)
    monkeypatch.setattr(tracking, "search_logged_models", search)
    payload = {
        "experiment_ids": [hidden, *requested_ids] if grant != "wildcard" else requested_ids,
        "filter": options["filter_string"],
        "max_results": options["max_results"],
        "order_by": options["order_by"],
        "page_token": old_page.token,
    }
    client, headers = logged_model_search_client
    response = client.post(
        f"{prefix}/2.0/mlflow/logged-models/search", json=payload, headers=headers
    )
    assert response.status_code == 400
    assert response.json["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "Experiment IDs in the page token do not match" in response.json["message"]
    assert "models" not in response.json
    search.assert_called_once_with(
        experiment_ids=[second, first],
        page_token=old_page.token,
        datasets=None,
        **options,
    )


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize(
    "change",
    [
        "deny",
        "model-tier-deny",
        "caller-ids",
        "caller-order",
        "filter",
        "order-by",
        "invalid-token",
        "foreign-token",
    ],
)
def test_logged_model_token_validation_uses_effective_scope_without_retry(
    large_scope_search_setup, logged_model_search_client, monkeypatch, prefix, change
):
    auth_store, tracking, _, role, _, _ = large_scope_search_setup
    first = tracking.create_experiment("first")
    second = tracking.create_experiment("second")
    hidden = tracking.create_experiment("hidden")
    for experiment_id, name in [(first, "a"), (second, "b"), (hidden, "0-hidden")]:
        tracking.create_logged_model(experiment_id, name=name)
        tracking.create_logged_model(experiment_id, name=f"{name}-2")
    permissions = {
        experiment_id: auth_store.add_role_permission(
            role.id, "experiment", experiment_id, READ.name
        )
        for experiment_id in (first, second)
    }
    ids = [first, second]
    options = {
        "filter_string": "name != 'excluded'",
        "order_by": [
            {
                "field_name": "name",
                "ascending": True,
                "dataset_name": None,
                "dataset_digest": None,
            }
        ],
        "max_results": 1,
    }
    old_page = tracking.search_logged_models(
        experiment_ids=[hidden, hidden] if change == "foreign-token" else ids, **options
    )
    assert old_page.token
    payload = {
        "experiment_ids": ids,
        "filter": options["filter_string"],
        "order_by": options["order_by"],
        "max_results": 1,
        "page_token": old_page.token,
    }
    expected_ids = {first, second}
    if change == "deny":
        auth_store.update_role_permission(permissions[second].id, DENY.name)
        expected_ids = {first}
    elif change == "model-tier-deny":
        auth_store.add_role_permission(role.id, "logged_model", "*", DENY.name)
        expected_ids = set()
    elif change == "caller-ids":
        payload["experiment_ids"] = [first, first]
        expected_ids = {first}
    elif change == "caller-order":
        payload["experiment_ids"] = [second, second, first, first]
    elif change == "filter":
        payload["filter"] = "name = 'a'"
    elif change == "order-by":
        payload["order_by"] = [{**options["order_by"][0], "ascending": False}]
    elif change == "invalid-token":
        payload["page_token"] = base64.b64encode(b"not-json").decode("ascii")  # notsecret
    search = Mock(wraps=tracking.search_logged_models)
    monkeypatch.setattr(tracking, "search_logged_models", search)
    client, headers = logged_model_search_client
    response = client.post(
        f"{prefix}/2.0/mlflow/logged-models/search", json=payload, headers=headers
    )
    assert response.status_code == 400
    assert response.json["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "models" not in response.json
    search.assert_called_once()
    for call in search.call_args_list:
        assert set(call.kwargs["experiment_ids"]) == expected_ids
        assert call.kwargs["page_token"] == payload["page_token"]
        assert call.kwargs["filter_string"] == payload["filter"]
        assert call.kwargs["order_by"] == payload["order_by"]


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("error_kind", ["native", "rest"])
def test_logged_model_remote_token_mismatch_is_returned_without_retry(
    large_scope_search_setup, logged_model_search_client, monkeypatch, prefix, error_kind
):
    auth_store, _, _, role, _, _ = large_scope_search_setup
    for experiment_id in ("1", "2"):
        auth_store.add_role_permission(role.id, "experiment", experiment_id, READ.name)
    auth_store.add_role_permission(role.id, "experiment", "3", DENY.name)
    mismatch = MlflowException.invalid_parameter_value(
        "Experiment IDs in the page token do not match the requested experiment IDs."
    )
    if error_kind == "rest":
        mismatch = RestException({"error_code": mismatch.error_code, "message": mismatch.message})
    rpc = Mock(side_effect=mismatch)
    remote = TrackingRestStore(lambda: MlflowHostCreds("https://example.com"))
    monkeypatch.setattr(remote, "_call_endpoint", rpc)
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: remote)
    payload = {
        "experiment_ids": ["3", "2", "+2", "1", "+1", "2", "+3"],
        "page_token": "opaque-legacy-token",
        "max_results": 2,
        "filter": "name != 'excluded'",
        "datasets": [{"dataset_name": "eval", "dataset_digest": "digest"}],
        "order_by": [{"field_name": "name", "ascending": True}],
    }
    client, headers = logged_model_search_client
    response = client.post(
        f"{prefix}/2.0/mlflow/logged-models/search", json=payload, headers=headers
    )
    assert response.status_code == 400
    assert response.json["error_code"] == "INVALID_PARAMETER_VALUE"
    assert "Experiment IDs in the page token do not match" in response.json["message"]
    rpc.assert_called_once()
    assert rpc.call_args.args[0] == SearchLoggedModels
    query = json.loads(rpc.call_args.args[1])
    assert query == {**payload, "experiment_ids": ["2", "1"]}


@pytest.mark.parametrize("outcome", ["unrelated", "wrong-code", "no-token", "admin", "unavailable"])
def test_logged_model_search_returns_backend_errors_without_retry(
    large_scope_search_setup, logged_model_search_client, monkeypatch, outcome
):
    auth_store, tracking, _, role, username, _ = large_scope_search_setup
    auth_store.add_role_permission(role.id, "experiment", "1", READ.name)
    mismatch_message = "Experiment IDs in the page token do not match the requested experiment IDs."
    error = MlflowException.invalid_parameter_value(mismatch_message)
    expected_status = 400
    if outcome == "unrelated":
        error = MlflowException.invalid_parameter_value("Invalid page token")
    elif outcome == "wrong-code":
        error = MlflowException(mismatch_message, INTERNAL_ERROR)
        expected_status = 500
    elif outcome == "admin":
        auth_store.update_user(username, is_admin=True)
    if outcome == "unavailable":
        error = MlflowException("Database unavailable", TEMPORARILY_UNAVAILABLE)
        expected_status = 503
    search = Mock(side_effect=error)
    monkeypatch.setattr(tracking, "search_logged_models", search)
    payload = {"experiment_ids": ["1", "+1"], "page_token": "opaque-token", "max_results": 1}
    if outcome == "no-token":
        payload.pop("page_token")
    client, headers = logged_model_search_client
    response = client.post("/api/2.0/mlflow/logged-models/search", json=payload, headers=headers)
    assert response.status_code == expected_status
    search.assert_called_once()
    assert search.call_args_list[0].kwargs["experiment_ids"] == (
        payload["experiment_ids"] if outcome == "admin" else ["1"]
    )
    for call in search.call_args_list:
        assert call.kwargs["page_token"] == payload.get("page_token")


@pytest.fixture
def experiment_search_client(large_scope_search_setup):
    _, _, _, _, username, password = large_scope_search_setup
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    for prefix in ("/api", "/ajax-api"):
        app.add_url_rule(
            f"{prefix}/2.0/mlflow/experiments/search",
            endpoint=prefix,
            view_func=handlers._search_experiments,
            methods=["GET", "POST"],
        )
    headers = {
        "Authorization": Authorization(
            "basic", {"username": username, "password": password}
        ).to_header()
    }
    return app.test_client(), headers


@pytest.mark.parametrize("backend", ["sql", "file"])
@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("max_results", [1, 2])
@pytest.mark.parametrize(
    "grant",
    ["named", "wildcard", "wildcard-deny", "deny-named-read", "none", "use", "manage"],
)
def test_experiment_search_scopes_before_storage(
    large_scope_search_setup,
    experiment_search_client,
    tmp_path,
    monkeypatch,
    backend,
    prefix,
    method,
    max_results,
    grant,
):
    auth_store, tracking, _, role, username, _ = large_scope_search_setup
    if backend == "file":
        monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "false")
        monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
        tracking = FileStore(str(tmp_path / "experiment-file-store"))
        role = auth_store.create_role(name="file-search", workspace=DEFAULT_WORKSPACE_NAME)
        auth_store.assign_role_to_user(auth_store.get_user(username).id, role.id)
        for module in (auth, handlers):
            monkeypatch.setattr(module, "_get_tracking_store", lambda: tracking)
    ids = [tracking.create_experiment(f"scope-{name}") for name in ("a", "b", "c", "d")]
    if backend == "sql":
        with workspace_context.WorkspaceContext("team-b"):
            outside = tracking.create_experiment("scope-outside")
        auth_store.add_role_permission(role.id, "experiment", outside, READ.name)
    if grant in {"named", "deny-named-read"}:
        for experiment_id in (ids[1], ids[3]):
            auth_store.add_role_permission(role.id, "experiment", experiment_id, READ.name)
    if grant in {"wildcard", "wildcard-deny", "deny-named-read"}:
        auth_store.add_role_permission(
            role.id, "experiment", "*", DENY.name if grant == "deny-named-read" else READ.name
        )
    if grant == "wildcard-deny":
        auth_store.add_role_permission(role.id, "experiment", ids[2], DENY.name)
    if grant in {"use", "manage"}:
        auth_store.add_role_permission(
            role.id, "workspace", "*", USE.name if grant == "use" else MANAGE.name
        )
    expected = {
        "named": [ids[1], ids[3]],
        "wildcard": ids,
        "wildcard-deny": [ids[0], ids[1], ids[3]],
        "deny-named-read": [],
        "none": [],
        "use": [],
        "manage": ids,
    }[grant]
    search = Mock(wraps=tracking.search_experiments)
    monkeypatch.setattr(tracking, "search_experiments", search)
    client, headers = experiment_search_client
    params = {"filter": "name LIKE 'scope%'", "max_results": max_results, "order_by": ["name ASC"]}
    pages = []
    for _ in range(6):
        kwargs = {"query_string": params} if method == "GET" else {"json": params}
        response = client.open(
            f"{prefix}/2.0/mlflow/experiments/search", method=method, headers=headers, **kwargs
        )
        assert response.status_code == 200
        pages.append([item["experiment_id"] for item in response.json.get("experiments", [])])
        if not (token := response.json.get("next_page_token")):
            break
        params["page_token"] = token
    else:
        pytest.fail("Experiment pagination did not terminate")
    assert [value for page in pages for value in page] == expected
    assert pages == [
        expected[i : i + max_results] for i in range(0, len(expected), max_results)
    ] or (not expected and pages == [[]])
    search.assert_called()
    for call in search.call_args_list:
        parsed = SearchExperimentsUtils.parse_search_filter(call.kwargs["filter_string"])
        assert parsed[0]["key"] == "name"
        assert call.kwargs["order_by"] == ["name ASC"]
        if grant in {"wildcard", "manage"}:
            assert len(parsed) == 1
        else:
            assert any(clause["key"] == "experiment_id" for clause in parsed)


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("view_type", [ViewType.ACTIVE_ONLY, ViewType.DELETED_ONLY, ViewType.ALL])
def test_experiment_search_scope_preserves_caller_filter_and_lifecycle(
    large_scope_search_setup, experiment_search_client, monkeypatch, prefix, method, view_type
):
    auth_store, tracking, _, role, _, _ = large_scope_search_setup
    first = tracking.create_experiment("first")
    second = tracking.create_experiment("second")
    denied = tracking.create_experiment("denied")
    tracking.delete_experiment(second)
    auth_store.add_role_permission(role.id, "experiment", "*", READ.name)
    auth_store.add_role_permission(role.id, "experiment", denied, DENY.name)
    caller_filter = f"experiment_id IN ('{second}', '{denied}')"
    search = Mock(wraps=tracking.search_experiments)
    monkeypatch.setattr(tracking, "search_experiments", search)
    client, headers = experiment_search_client
    params = {"filter": caller_filter, "view_type": view_type, "max_results": 10}
    kwargs = {"query_string": params} if method == "GET" else {"json": params}
    response = client.open(
        f"{prefix}/2.0/mlflow/experiments/search", method=method, headers=headers, **kwargs
    )
    assert response.status_code == 200
    result_ids = [item["experiment_id"] for item in response.json.get("experiments", [])]
    assert result_ids == ([] if view_type == ViewType.ACTIVE_ONLY else [second])
    assert first not in result_ids
    search.assert_called_once()
    assert search.call_args.kwargs["filter_string"] == (
        f"{caller_filter} AND experiment_id NOT IN ('{denied}')"
    )
    assert search.call_args.kwargs["view_type"] == view_type


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize(
    "caller_filter",
    ["name = 'visible' -- ignored", "name = 'visible' OR name = 'hidden'", "bad = 'x'"],
)
def test_experiment_search_scope_validates_filter_before_storage(
    large_scope_search_setup, experiment_search_client, monkeypatch, method, caller_filter
):
    _, tracking, _, _, _, _ = large_scope_search_setup
    search = Mock(wraps=tracking.search_experiments)
    monkeypatch.setattr(tracking, "search_experiments", search)
    client, headers = experiment_search_client
    params = {"filter": caller_filter}
    kwargs = {"query_string": params} if method == "GET" else {"json": params}
    response = client.open(
        "/api/2.0/mlflow/experiments/search", method=method, headers=headers, **kwargs
    )
    assert response.status_code == 400
    assert response.json["error_code"] == "INVALID_PARAMETER_VALUE"
    search.assert_not_called()


def test_experiment_search_does_not_retry_without_scope_on_store_error(
    large_scope_search_setup, experiment_search_client, monkeypatch
):
    auth_store, tracking, _, role, _, _ = large_scope_search_setup
    visible = tracking.create_experiment("visible")
    auth_store.add_role_permission(role.id, "experiment", visible, READ.name)
    search = Mock(side_effect=MlflowException.invalid_parameter_value("Store scope limit exceeded"))
    monkeypatch.setattr(tracking, "search_experiments", search)
    client, headers = experiment_search_client
    response = client.get(
        "/api/2.0/mlflow/experiments/search", query_string={"max_results": 1}, headers=headers
    )
    assert response.status_code == 400
    assert response.json["error_code"] == "INVALID_PARAMETER_VALUE"
    search.assert_called_once()
    assert search.call_args.kwargs["filter_string"] == f"experiment_id IN ('{visible}')"


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("mode", ["admin", "default-read", "default-read-deny"])
def test_experiment_search_preserves_admin_and_default_permissions(
    large_scope_search_setup, experiment_search_client, tmp_path, monkeypatch, method, mode
):
    auth_store, _, _, _, username, password = large_scope_search_setup
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "false")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking = FileStore(str(tmp_path / "default-experiment-search"))
    for module in (auth, handlers):
        monkeypatch.setattr(module, "_get_tracking_store", lambda: tracking)
    first = tracking.create_experiment("scope-first")
    second = tracking.create_experiment("scope-second")
    client, headers = experiment_search_client
    if mode == "admin":
        auth_store.create_user("experiment-admin", password, is_admin=True)
        headers = {
            "Authorization": Authorization(
                "basic", {"username": "experiment-admin", "password": password}
            ).to_header()
        }
    else:
        monkeypatch.setattr(
            auth, "auth_config", auth.auth_config._replace(default_permission=READ.name)
        )
        if mode == "default-read-deny":
            role = auth_store.create_role(name="default-deny", workspace=DEFAULT_WORKSPACE_NAME)
            auth_store.assign_role_to_user(auth_store.get_user(username).id, role.id)
            auth_store.add_role_permission(role.id, "experiment", second, DENY.name)
    search = Mock(wraps=tracking.search_experiments)
    monkeypatch.setattr(tracking, "search_experiments", search)
    params = {"filter": "name LIKE 'scope%'", "max_results": 10, "order_by": ["name ASC"]}
    kwargs = {"query_string": params} if method == "GET" else {"json": params}
    response = client.open(
        "/api/2.0/mlflow/experiments/search", method=method, headers=headers, **kwargs
    )
    assert response.status_code == 200
    assert [item["experiment_id"] for item in response.json.get("experiments", [])] == (
        [first] if mode == "default-read-deny" else [first, second]
    )
    search.assert_called_once()
    assert search.call_args.kwargs["filter_string"] == (
        f"name LIKE 'scope%' AND experiment_id NOT IN ('{second}')"
        if mode == "default-read-deny"
        else "name LIKE 'scope%'"
    )


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("empty", [True, False])
def test_experiment_search_preserves_backend_page_without_additional_queries(
    large_scope_search_setup, experiment_search_client, monkeypatch, method, empty
):
    auth_store, tracking, _, role, _, _ = large_scope_search_setup
    visible = tracking.create_experiment("visible")
    auth_store.add_role_permission(role.id, "experiment", visible, READ.name)
    token = "opaque-backend-continuation"
    rows = [] if empty else [tracking.get_experiment(visible)]
    search = Mock(return_value=PagedList(rows, token))
    fetch = Mock(wraps=tracking.get_experiment)
    monkeypatch.setattr(tracking, "search_experiments", search)
    monkeypatch.setattr(tracking, "get_experiment", fetch)
    client, headers = experiment_search_client
    params = {"filter": "name != 'Default'", "max_results": 2, "order_by": ["name ASC"]}
    kwargs = {"query_string": params} if method == "GET" else {"json": params}
    response = client.open(
        "/api/2.0/mlflow/experiments/search", method=method, headers=headers, **kwargs
    )
    assert response.status_code == 200
    assert [item["experiment_id"] for item in response.json.get("experiments", [])] == (
        [] if empty else [visible]
    )
    assert response.json["next_page_token"] == token
    search.assert_called_once()
    fetch.assert_not_called()
    assert search.call_args.kwargs["filter_string"] == (
        f"name != 'Default' AND experiment_id IN ('{visible}')"
    )
    assert search.call_args.kwargs["order_by"] == ["name ASC"]
    assert search.call_args.kwargs["max_results"] == 2


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("alias", ["plus", "leading-zero", "whitespace"])
def test_experiment_search_does_not_reinterpret_grant_id_aliases(
    large_scope_search_setup, experiment_search_client, monkeypatch, method, alias
):
    auth_store, tracking, _, role, _, _ = large_scope_search_setup
    visible = tracking.create_experiment("visible")
    pattern = {
        "plus": f"+{visible}",
        "leading-zero": f"0{visible}",
        "whitespace": f" {visible} ",
    }[alias]
    auth_store.add_role_permission(role.id, "experiment", "*", READ.name)
    auth_store.add_role_permission(role.id, "experiment", pattern, DENY.name)
    search = Mock(wraps=tracking.search_experiments)
    monkeypatch.setattr(tracking, "search_experiments", search)
    client, headers = experiment_search_client
    params = {"filter": "name = 'visible'", "max_results": 10}
    kwargs = {"query_string": params} if method == "GET" else {"json": params}
    response = client.open(
        "/api/2.0/mlflow/experiments/search", method=method, headers=headers, **kwargs
    )
    assert response.status_code == 200
    assert [item["experiment_id"] for item in response.json.get("experiments", [])] == [visible]
    search.assert_called_once()
    assert search.call_args.kwargs["filter_string"] == "name = 'visible'"


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("grant", ["named-read", "wildcard-deny", "only-out-of-range"])
@pytest.mark.parametrize("invalid_id", [str(2**63), "9" * 255])
def test_experiment_search_ignores_out_of_range_grant_ids(
    large_scope_search_setup, experiment_search_client, monkeypatch, method, grant, invalid_id
):
    auth_store, tracking, _, role, _, _ = large_scope_search_setup
    visible = tracking.create_experiment("visible")
    if grant == "named-read":
        auth_store.add_role_permission(role.id, "experiment", visible, READ.name)
    elif grant == "wildcard-deny":
        auth_store.add_role_permission(role.id, "experiment", "*", READ.name)
    auth_store.add_role_permission(
        role.id, "experiment", invalid_id, DENY.name if grant == "wildcard-deny" else READ.name
    )
    search = Mock(wraps=tracking.search_experiments)
    monkeypatch.setattr(tracking, "search_experiments", search)
    params = {"filter": "name = 'visible'", "max_results": 10}
    kwargs = {"query_string": params} if method == "GET" else {"json": params}
    client, headers = experiment_search_client
    response = client.open(
        "/api/2.0/mlflow/experiments/search", method=method, headers=headers, **kwargs
    )
    assert response.status_code == 200
    assert [row["experiment_id"] for row in response.json.get("experiments", [])] == (
        [] if grant == "only-out-of-range" else [visible]
    )
    search.assert_called_once()
    effective_filter = search.call_args.kwargs["filter_string"]
    assert invalid_id not in effective_filter
    assert (
        effective_filter
        == {
            "named-read": f"name = 'visible' AND experiment_id IN ('{visible}')",
            "wildcard-deny": "name = 'visible'",
            "only-out-of-range": (
                "name = 'visible' AND experiment_id = '0' AND experiment_id != '0'"
            ),
        }[grant]
    )


def test_experiment_search_keeps_largest_signed_integer_id(
    large_scope_search_setup, experiment_search_client, tmp_path, monkeypatch
):
    auth_store, _, _, _, username, _ = large_scope_search_setup
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "false")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking = FileStore(str(tmp_path / "largest-experiment-id"))
    for module in (auth, handlers):
        monkeypatch.setattr(module, "_get_tracking_store", lambda: tracking)
    role = auth_store.create_role(name="largest-id", workspace=DEFAULT_WORKSPACE_NAME)
    auth_store.assign_role_to_user(auth_store.get_user(username).id, role.id)
    largest_id = str(2**63 - 1)
    generate_id = Mock(return_value=int(largest_id))
    monkeypatch.setattr("mlflow.store.tracking.file_store._generate_unique_integer_id", generate_id)
    assert tracking.create_experiment("largest-id") == largest_id
    generate_id.assert_called_once()
    auth_store.add_role_permission(role.id, "experiment", largest_id, READ.name)
    client, headers = experiment_search_client
    response = client.get(
        "/api/2.0/mlflow/experiments/search", query_string={"max_results": 10}, headers=headers
    )
    assert response.status_code == 200
    assert [row["experiment_id"] for row in response.json["experiments"]] == [largest_id]


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("collection", ["servers", "endpoints"])
@pytest.mark.parametrize("grant", ["named", "wildcard-deny", "none", "unrestricted"])
@pytest.mark.parametrize("invalid", ["attribute", "comment", "or"])
def test_mcp_invalid_search_filter_preserves_json_error_contract(
    large_scope_search_setup, monkeypatch, prefix, collection, grant, invalid
):
    auth_store, tracking, _, role, username, _ = large_scope_search_setup
    if grant == "named":
        auth_store.add_role_permission(role.id, "mcp_server", "com.example/visible", READ.name)
    elif grant in {"wildcard-deny", "unrestricted"}:
        auth_store.add_role_permission(role.id, "mcp_server", "*", READ.name)
        if grant == "wildcard-deny":
            auth_store.add_role_permission(role.id, "mcp_server", "com.example/hidden", DENY.name)
    user = auth_store.get_user(username)
    monkeypatch.setattr(auth, "_authenticate_fastapi_request", lambda _: user)
    monkeypatch.setattr(
        auth, "resolve_workspace_for_request_if_enabled", lambda *_: SimpleNamespace(name="team-a")
    )
    app = _create_registry_fastapi_app()

    @app.middleware("http")
    async def workspace_middleware(request, call_next):
        with workspace_context.WorkspaceContext("team-a"):
            return await call_next(request)

    auth.add_fastapi_permission_middleware(app)
    path = f"{prefix}/3.0/mlflow/mcp-servers"
    method = "search_mcp_servers"
    key = "name"
    if collection == "endpoints":
        path += "/endpoints"
        method = "search_mcp_access_endpoints"
        key = "server_name"
    caller_filter = {
        "attribute": "invalid_attribute = 'value'",
        "comment": f"{key} = 'x' -- ignored",
        "or": f"{key} = 'x' OR {key} = 'y'",
    }[invalid]
    search = Mock(wraps=getattr(tracking, method))
    monkeypatch.setattr(tracking, method, search)
    response = TestClient(app).get(path, params={"filter_string": caller_filter})

    assert response.status_code == 400
    assert response.headers["content-type"] == "application/json"
    assert response.json()["error_code"] == "INVALID_PARAMETER_VALUE"
    assert response.json()["message"]
    with pytest.raises(RestException, match="INVALID_PARAMETER_VALUE") as client_error:
        verify_rest_response(response, path)
    assert client_error.value.error_code == "INVALID_PARAMETER_VALUE"
    if grant == "unrestricted":
        search.assert_called_once()
        assert search.call_args.kwargs["filter_string"] == caller_filter
    else:
        search.assert_not_called()


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("outcome", ["denied", "exception"])
def test_mcp_validator_responses_unchanged_by_scoper_error_serialization(
    large_scope_search_setup, monkeypatch, prefix, outcome
):
    auth_store, tracking, _, _, username, _ = large_scope_search_setup
    user = auth_store.get_user(username)
    monkeypatch.setattr(auth, "_authenticate_fastapi_request", lambda _: user)
    monkeypatch.setattr(
        auth, "resolve_workspace_for_request_if_enabled", lambda *_: SimpleNamespace(name="team-a")
    )

    async def validator(username, request):
        if outcome == "exception":
            raise MlflowException.invalid_parameter_value("Invalid request from validator")
        return False

    monkeypatch.setattr(auth, "_find_fastapi_validator", lambda *_: validator)
    scoper = Mock(wraps=auth._scope_mcp_server_search_query)
    monkeypatch.setattr(auth, "_scope_mcp_server_search_query", scoper)
    search = Mock(wraps=tracking.search_mcp_servers)
    monkeypatch.setattr(tracking, "search_mcp_servers", search)
    app = _create_registry_fastapi_app()
    auth.add_fastapi_permission_middleware(app)
    response = TestClient(app).get(f"{prefix}/3.0/mlflow/mcp-servers")

    assert response.status_code == (403 if outcome == "denied" else 400)
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == (
        "Permission denied" if outcome == "denied" else "Invalid request from validator"
    )
    scoper.assert_not_called()
    search.assert_not_called()


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
def test_mcp_search_resolves_permissions_once_per_server(
    large_scope_search_setup, monkeypatch, prefix
):
    auth_store, tracking, _, role, username, _ = large_scope_search_setup
    auth_store.add_role_permission(role.id, "mcp_server", "*", READ.name)
    names = [f"com.example/visible-{index:02}" for index in range(20)]
    for name in names:
        tracking.create_mcp_server(name=name)
    user = auth_store.get_user(username)
    monkeypatch.setattr(auth, "_authenticate_fastapi_request", lambda _: user)
    monkeypatch.setattr(
        auth, "resolve_workspace_for_request_if_enabled", lambda *_: SimpleNamespace(name="team-a")
    )
    app = _create_registry_fastapi_app()

    @app.middleware("http")
    async def workspace_middleware(request, call_next):
        with workspace_context.WorkspaceContext("team-a"):
            return await call_next(request)

    auth.add_fastapi_permission_middleware(app)
    permission = Mock(wraps=auth._get_mcp_server_permission)
    grants = Mock(wraps=auth_store.list_grants)
    monkeypatch.setattr(auth, "_get_mcp_server_permission", permission)
    monkeypatch.setattr(auth_store, "list_grants", grants)
    response = TestClient(app).get(f"{prefix}/3.0/mlflow/mcp-servers", params={"max_results": 20})
    assert response.status_code == 200
    servers = response.json()["mcp_servers"]
    assert sorted(row["name"] for row in servers) == names
    assert all(row["allowed_actions"] == [] for row in servers)
    assert permission.call_count == len(names)
    assert sorted(call.args for call in permission.call_args_list) == [
        (name, username) for name in names
    ]
    assert grants.call_count == len(names) + 1


@pytest.mark.parametrize("collection", ["experiments", "servers", "endpoints"])
@pytest.mark.parametrize("grant", ["named", "wildcard-deny"])
@pytest.mark.parametrize("limit", ["parameters", "parser", "native"])
@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
def test_scoped_search_reports_backend_limits_without_retry(
    large_scope_search_setup,
    experiment_search_client,
    monkeypatch,
    collection,
    grant,
    limit,
    prefix,
):
    auth_store, tracking, _, role, username, _ = large_scope_search_setup
    resource_type = "experiment" if collection == "experiments" else "mcp_server"
    scope_size = 3400 if limit == "parser" else 500
    values = [
        str(index + 10000) if collection == "experiments" else f"com.example/scope-{index}"
        for index in range(scope_size)
    ]
    if grant == "wildcard-deny":
        auth_store.add_role_permission(role.id, resource_type, "*", READ.name)
    with auth_store.ManagedSessionMaker(read_only=False) as session:
        session.add_all([
            SqlRolePermission(
                role_id=role.id,
                resource_type=resource_type,
                resource_pattern=value,
                permission=DENY.name if grant == "wildcard-deny" else READ.name,
            )
            for value in values
        ])
    scope_key = {"experiments": "experiment_id", "servers": "name", "endpoints": "server_name"}[
        collection
    ]
    caller_filter = f"{scope_key} IN ({', '.join(repr(value) for value in values[:500])})"
    statements = []

    def guard_parameters(conn, cursor, statement, parameters, context, executemany):
        if not executemany and len(parameters) > 999:
            statements.append((statement, parameters))
            # Simulate a historical SQLite budget on Python versions without setlimit().
            raise OperationalError(
                statement, parameters, sqlite3.OperationalError("too many SQL variables")
            )

    if limit == "parameters":
        event.listen(tracking.engine, "before_cursor_execute", guard_parameters)
    method_name = {
        "experiments": "search_experiments",
        "servers": "search_mcp_servers",
        "endpoints": "search_mcp_access_endpoints",
    }[collection]
    search = Mock(wraps=getattr(tracking, method_name))
    monkeypatch.setattr(tracking, method_name, search)
    try:
        if collection == "experiments":
            client, headers = experiment_search_client
            response = client.get(
                f"{prefix}/2.0/mlflow/experiments/search",
                query_string={"filter": caller_filter, "max_results": 1},
                headers=headers,
            )
            body = response.json
        else:
            user = auth_store.get_user(username)
            monkeypatch.setattr(auth, "_authenticate_fastapi_request", lambda _request: user)
            monkeypatch.setattr(
                auth,
                "resolve_workspace_for_request_if_enabled",
                lambda *_args: SimpleNamespace(name="team-a"),
            )
            app = _create_registry_fastapi_app()

            @app.middleware("http")
            async def workspace_middleware(request, call_next):
                with workspace_context.WorkspaceContext("team-a"):
                    return await call_next(request)

            auth.add_fastapi_permission_middleware(app)
            path = f"{prefix}/3.0/mlflow/mcp-servers"
            if collection == "endpoints":
                path += "/endpoints"
            response = TestClient(app).get(
                path, params={"filter_string": caller_filter, "max_results": 1}
            )
            body = response.json()
    finally:
        if limit == "parameters":
            event.remove(tracking.engine, "before_cursor_execute", guard_parameters)

    if limit == "native":
        assert response.status_code == 200
        key = {
            "experiments": "experiments",
            "servers": "mcp_servers",
            "endpoints": "mcp_access_endpoints",
        }[collection]
        assert body.get(key, []) == []
    else:
        assert response.status_code == 400
        assert body["error_code"] == "INVALID_PARAMETER_VALUE"
        assert "authorization scope exceeds a backend limit" in body["message"]
        assert values[0] not in body["message"]
    search.assert_called_once()
    effective_filter = search.call_args.kwargs["filter_string"]
    assert effective_filter.startswith(f"{caller_filter} AND ")
    assert ("NOT IN" in effective_filter) == (grant == "wildcard-deny")
    if limit == "parameters":
        assert len(statements) == 1
        assert len(statements[0][1]) > 999


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("infos", [False, True])
@pytest.mark.parametrize("valid_grant", [False, True])
@pytest.mark.parametrize("invalid_id", [str(2**63), "9" * 255])
def test_batch_trace_scope_excludes_oversized_stored_grants(
    large_scope_search_setup, monkeypatch, prefix, infos, valid_grant, invalid_id
):
    auth_store, tracking, _, role, username, password = large_scope_search_setup
    readable = tracking.create_experiment("readable")
    tracking.log_spans(readable, [create_test_span(trace_id="tr-readable")])
    if valid_grant:
        auth_store.add_role_permission(role.id, "experiment", readable, READ.name)
    auth_store.add_role_permission(role.id, "experiment", invalid_id, READ.name)
    method_name = "batch_get_trace_infos" if infos else "batch_get_traces"
    batch = Mock(wraps=getattr(tracking, method_name))
    monkeypatch.setattr(tracking, method_name, batch)
    path = f"{prefix}/3.0/mlflow/traces/" + ("batchGetInfos" if infos else "batchGet")
    method = "POST" if infos else "GET"
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    app.add_url_rule(
        path,
        view_func=handlers._batch_get_trace_infos if infos else handlers._batch_get_traces,
        methods=[method],
    )
    response = app.test_client().open(
        path,
        method=method,
        headers={
            "Authorization": Authorization(
                "basic", {"username": username, "password": password}
            ).to_header()
        },
        **{"json" if infos else "query_string": {"trace_ids": ["tr-readable"]}},
    )
    assert response.status_code == 200
    batch.assert_called_once()
    assert batch.call_args.kwargs["experiment_ids"] == ([readable] if valid_grant else [])
    assert len(response.json.get("trace_infos" if infos else "traces", [])) == int(valid_grant)


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize(
    ("infos", "caller_scope"),
    [
        (False, "mixed"),
        (False, "denied"),
        (False, "aliases"),
        (True, "mixed"),
        (True, "denied"),
        (True, "aliases"),
        (True, "empty"),
    ],
)
def test_batch_trace_wildcard_scope_applies_named_deny_before_storage(
    large_scope_search_setup, monkeypatch, prefix, infos, caller_scope
):
    auth_store, tracking, _, role, username, password = large_scope_search_setup
    readable = tracking.create_experiment("readable")
    denied = tracking.create_experiment("denied")
    for experiment_id, trace_id in ((readable, "tr-readable"), (denied, "tr-denied")):
        tracking.log_spans(experiment_id, [create_test_span(trace_id=trace_id)])
    auth_store.add_role_permission(role.id, "experiment", "*", READ.name)
    auth_store.add_role_permission(role.id, "experiment", denied, DENY.name)
    method_name = "batch_get_trace_infos" if infos else "batch_get_traces"
    original = getattr(tracking, method_name)
    fetched = []

    def capture(*args, **kwargs):
        rows = original(*args, **kwargs)
        fetched.extend(rows)
        return rows

    batch = Mock(side_effect=capture)
    monkeypatch.setattr(tracking, method_name, batch)
    path = f"{prefix}/3.0/mlflow/traces/" + ("batchGetInfos" if infos else "batchGet")
    method = "POST" if infos else "GET"
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    app.add_url_rule(
        path,
        view_func=handlers._batch_get_trace_infos if infos else handlers._batch_get_traces,
        methods=[method],
    )
    requested = {
        "mixed": [readable, denied],
        "denied": [denied],
        "aliases": [f"+{denied}", f"0{readable}", readable, f" {denied} "],
        "empty": [],
    }[caller_scope]
    payload = {"trace_ids": ["tr-readable", "tr-denied"], "experiment_ids": requested}
    response = app.test_client().open(
        path,
        method=method,
        headers={
            "Authorization": Authorization(
                "basic", {"username": username, "password": password}
            ).to_header()
        },
        **{"json" if infos else "query_string": payload},
    )
    assert response.status_code == 200
    batch.assert_called_once()
    expected = [readable] if caller_scope in {"mixed", "aliases"} else []
    assert batch.call_args.kwargs["experiment_ids"] == expected
    assert len(fetched) == len(expected)
    assert len(response.json.get("trace_infos" if infos else "traces", [])) == len(expected)


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("row_count", [0, 1, 2])
@pytest.mark.parametrize("repeat_token", [False, True])
def test_logged_model_search_preserves_backend_rows_and_opaque_token(
    large_scope_search_setup,
    logged_model_search_client,
    monkeypatch,
    prefix,
    row_count,
    repeat_token,
):
    auth_store, tracking, _, role, _, _ = large_scope_search_setup
    readable = tracking.create_experiment("readable")
    auth_store.add_role_permission(role.id, "experiment", readable, READ.name)
    rows = [tracking.create_logged_model(readable, name=name) for name in ("a", "b")][:row_count]
    request_token = "opaque-request-token"
    token = request_token if repeat_token else "opaque-backend-token"
    search = Mock(return_value=PagedList(rows, token))
    predicate = Mock(wraps=auth._role_based_read_predicate)
    monkeypatch.setattr(tracking, "search_logged_models", search)
    monkeypatch.setattr(auth, "_role_based_read_predicate", predicate)
    client, headers = logged_model_search_client
    response = client.post(
        f"{prefix}/2.0/mlflow/logged-models/search",
        json={"experiment_ids": [readable], "max_results": 2, "page_token": request_token},
        headers=headers,
    )
    assert response.status_code == 200
    assert [row["info"]["model_id"] for row in response.json.get("models", [])] == [
        model.model_id for model in rows
    ]
    assert response.json["next_page_token"] == token
    search.assert_called_once_with(
        experiment_ids=[readable],
        filter_string=None,
        datasets=None,
        max_results=2,
        order_by=None,
        page_token=request_token,
    )
    predicate.assert_called()
    assert predicate.call_count == 2


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("scope", ["selected", "alias", "empty"])
@pytest.mark.parametrize("max_results", [1, 2, 10])
@pytest.mark.parametrize("grant", ["named", "wildcard-deny"])
def test_logged_model_search_enforces_caller_scope_before_backend_pagination(
    large_scope_search_setup,
    logged_model_search_client,
    monkeypatch,
    prefix,
    scope,
    max_results,
    grant,
):
    auth_store, tracking, _, role, _, _ = large_scope_search_setup
    selected = tracking.create_experiment("selected")
    excluded = tracking.create_experiment("excluded-but-readable")
    denied = tracking.create_experiment("denied")
    with workspace_context.WorkspaceContext("team-b"):
        outside = tracking.create_experiment("outside")
        tracking.create_logged_model(outside, name="outside-model")
    auth_store.add_role_permission(role.id, "experiment", outside, READ.name)
    if grant == "named":
        for experiment_id in (selected, excluded):
            auth_store.add_role_permission(role.id, "experiment", experiment_id, READ.name)
    else:
        auth_store.add_role_permission(role.id, "experiment", "*", READ.name)
        auth_store.add_role_permission(role.id, "experiment", denied, DENY.name)
    for experiment_id, name in [
        (excluded, "a-excluded"),
        (selected, "b-visible"),
        (excluded, "c-excluded"),
        (selected, "d-visible"),
        (excluded, "e-excluded"),
        (denied, "z-denied"),
    ]:
        tracking.create_logged_model(experiment_id, name=name)
    backend_pages = []
    original = tracking.search_logged_models

    def search_page(**kwargs):
        page = original(**kwargs)
        token = page.token.decode() if isinstance(page.token, bytes) else page.token
        backend_pages.append(([model.name for model in page], token))
        return page

    search = Mock(side_effect=search_page)
    monkeypatch.setattr(tracking, "search_logged_models", search)
    caller_ids = {
        "selected": [selected, denied, outside],
        "alias": [f" +{selected} ", selected, denied, outside],
        "empty": [],
    }
    payload = {
        "experiment_ids": caller_ids[scope],
        "max_results": max_results,
        "filter": "name != 'excluded'",
        "order_by": [{"field_name": "name", "ascending": True}],
    }
    client, headers = logged_model_search_client
    served = []
    expected_ids = [] if scope == "empty" else [selected, outside]
    for index in range(3):
        response = client.post(
            f"{prefix}/2.0/mlflow/logged-models/search", json=payload, headers=headers
        )
        assert response.status_code == 200
        names = [row["info"]["name"] for row in response.json.get("models", [])]
        token = response.json.get("next_page_token") or None
        assert backend_pages[-1] == (names, token)
        assert search.call_count == index + 1
        assert search.call_args.kwargs["experiment_ids"] == expected_ids
        assert search.call_args.kwargs["filter_string"] == payload["filter"]
        assert search.call_args.kwargs["order_by"] == [
            {"field_name": "name", "ascending": True, "dataset_name": None, "dataset_digest": None}
        ]
        assert search.call_args.kwargs["max_results"] == max_results
        assert search.call_args.kwargs["page_token"] == payload.get("page_token")
        served.extend(names)
        if not token:
            break
        payload["page_token"] = token
    else:
        pytest.fail("Scoped pagination did not terminate")
    assert served == ([] if scope == "empty" else ["b-visible", "d-visible"])
    search.assert_called()


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("run_grant", [None, READ.name, DENY.name])
def test_logged_model_search_redacts_run_references_without_additional_queries(
    large_scope_search_setup, logged_model_search_client, monkeypatch, prefix, run_grant
):
    auth_store, _, _, role, _, _ = large_scope_search_setup
    auth_store.add_role_permission(role.id, "experiment", "1", READ.name)
    if run_grant:
        auth_store.add_role_permission(role.id, "run", "*", run_grant)
    page = SearchLoggedModels.Response(next_page_token="opaque-backend-token")
    model = page.models.add()
    model.info.model_id = "m-visible"
    model.info.experiment_id = "1"
    model.info.source_run_id = "run-visible"
    model.data.metrics.add(
        key="accuracy", value=0.9, timestamp=1, step=0, model_id="m-visible", run_id="run-visible"
    )
    remote = TrackingRestStore(lambda: MlflowHostCreds("https://example.com"))
    rpc = Mock(return_value=page)
    monkeypatch.setattr(remote, "_call_endpoint", rpc)
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: remote)
    client, headers = logged_model_search_client
    payload = {"experiment_ids": ["1"], "max_results": 1, "page_token": "opaque-request-token"}
    response = client.post(
        f"{prefix}/2.0/mlflow/logged-models/search", json=payload, headers=headers
    )
    assert response.status_code == 200
    assert response.json["next_page_token"] == page.next_page_token
    assert len(response.json["models"]) == 1
    returned = response.json["models"][0]
    assert returned["info"]["model_id"] == "m-visible"
    assert returned["data"]["metrics"][0]["model_id"] == "m-visible"
    assert returned["info"].get("source_run_id") == (
        None if run_grant == DENY.name else "run-visible"
    )
    assert returned["data"]["metrics"][0].get("run_id") == (
        None if run_grant == DENY.name else "run-visible"
    )
    rpc.assert_called_once()
    assert rpc.call_args.args[0] == SearchLoggedModels
    assert json.loads(rpc.call_args.args[1]) == payload


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("store_class", [TrackingRestStore, DatabricksTracingRestStore])
@pytest.mark.parametrize(
    "scope", ["experiment-denied", "model-tier-denied", "explicit-empty", "readable"]
)
@pytest.mark.parametrize("invalid", [None, "filter", "page-token"])
def test_logged_model_rest_scope_survives_downstream_validation(
    large_scope_search_setup,
    logged_model_search_client,
    monkeypatch,
    prefix,
    store_class,
    scope,
    invalid,
):
    auth_store, tracking, _, role, _, _ = large_scope_search_setup
    readable = tracking.create_experiment("readable")
    denied = tracking.create_experiment("denied")
    visible = tracking.create_logged_model(readable, name="visible")
    tracking.create_logged_model(denied, name="hidden")
    auth_store.add_role_permission(role.id, "experiment", readable, READ.name)
    if scope == "model-tier-denied":
        auth_store.add_role_permission(role.id, "logged_model", "*", DENY.name)
    path = f"{prefix}/2.0/mlflow/logged-models/search"
    backend = Flask("remote-backend")
    backend.add_url_rule(path, view_func=handlers._search_logged_models, methods=["POST"])
    backend_search = Mock(wraps=tracking.search_logged_models)
    monkeypatch.setattr(tracking, "search_logged_models", backend_search)

    def dispatch(api, body):
        with monkeypatch.context() as patch:
            patch.setattr(handlers, "_get_tracking_store", lambda: tracking)
            response = backend.test_client().post(path, json=json.loads(body))
        if response.status_code >= 400:
            raise RestException(response.json)
        result = api.Response()
        parse_dict(response.json, result)
        return result

    remote = store_class(lambda: MlflowHostCreds("https://remote-backend.invalid"))
    rpc = Mock(side_effect=dispatch)
    monkeypatch.setattr(remote, "_call_endpoint", rpc)
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda: remote)
    experiment_ids = {
        "experiment-denied": [denied],
        "model-tier-denied": [readable],
        "explicit-empty": [],
        "readable": [readable, denied],
    }[scope]
    payload = {"experiment_ids": experiment_ids, "max_results": 1}
    if invalid == "filter":
        payload["filter"] = "invalid filter"
    elif invalid == "page-token":
        payload["page_token"] = base64.b64encode(b"not-json").decode("ascii")  # notsecret
    client, headers = logged_model_search_client
    response = client.post(path, json=payload, headers=headers)

    assert response.status_code == (400 if invalid else 200)
    if invalid:
        assert response.json["error_code"] == "INVALID_PARAMETER_VALUE"
        assert "models" not in response.json
        assert "Missing value for required parameter" not in response.json["message"]
    else:
        assert [model["info"]["model_id"] for model in response.json.get("models", [])] == (
            [visible.model_id] if scope == "readable" else []
        )
        assert not response.json.get("next_page_token")
    expected_ids = [readable] if scope == "readable" else []
    rpc.assert_called_once()
    assert rpc.call_args.args[0] == SearchLoggedModels
    assert json.loads(rpc.call_args.args[1]) == {**payload, "experiment_ids": expected_ids}
    backend_search.assert_called_once()
    assert backend_search.call_args.kwargs["experiment_ids"] == expected_ids


@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("collection", ["experiments", "logged-models", "servers"])
def test_request_scopes_resolve_large_grant_sets_with_linear_work(
    large_scope_search_setup, monkeypatch, prefix, collection
):
    auth_store, tracking, _, role, username, password = large_scope_search_setup
    grant_count = 2000
    if collection == "servers":
        visible = "com.example/linear-visible"
        denied = "com.example/linear-denied"
        for name in (visible, denied):
            tracking.create_mcp_server(name)
        resource_type = "mcp_server"
        extra_ids = [f"com.example/unused-{index}" for index in range(grant_count)]
        user = auth_store.get_user(username)
        monkeypatch.setattr(auth, "_authenticate_fastapi_request", lambda _: user)
        monkeypatch.setattr(
            auth,
            "resolve_workspace_for_request_if_enabled",
            lambda *_: SimpleNamespace(name="team-a"),
        )
        app = _create_registry_fastapi_app()

        @app.middleware("http")
        async def workspace_middleware(request, call_next):
            with workspace_context.WorkspaceContext("team-a"):
                return await call_next(request)

        auth.add_fastapi_permission_middleware(app)
        client = TestClient(app)
        path = f"{prefix}/3.0/mlflow/mcp-servers"
        options = {"params": {"max_results": 1, "filter_string": "name LIKE 'com.example/linear%'"}}
        search_method = "search_mcp_servers"
    else:
        visible = tracking.create_experiment("linear-visible")
        denied = tracking.create_experiment("linear-denied")
        resource_type = "experiment"
        extra_ids = [str(100000 + index) for index in range(grant_count)]
        app = Flask(__name__)
        app.before_request(auth._before_request)
        app.after_request(auth._after_request)
        path = f"{prefix}/2.0/mlflow/{collection}/search"
        method = "POST" if collection == "logged-models" else "GET"
        handler = (
            handlers._search_logged_models
            if collection == "logged-models"
            else handlers._search_experiments
        )
        app.add_url_rule(path, view_func=handler, methods=[method])
        client = app.test_client()
        options = {
            "headers": {
                "Authorization": Authorization(
                    "basic", {"username": username, "password": password}
                ).to_header()
            }
        }
        if collection == "logged-models":
            for experiment_id in (visible, denied):
                tracking.create_logged_model(experiment_id, name="linear-model")
            options["json"] = {"experiment_ids": [visible, denied], "max_results": 1}
            search_method = "search_logged_models"
        else:
            options["query_string"] = {"max_results": 1, "filter": "name LIKE 'linear%'"}
            search_method = "search_experiments"
    deny_role = auth_store.create_role(name="linear-deny", workspace="team-a")
    auth_store.assign_role_to_user(auth_store.get_user(username).id, deny_role.id)
    auth_store.add_role_permission(deny_role.id, resource_type, denied, DENY.name)
    with auth_store.ManagedSessionMaker(read_only=False) as session:
        session.add_all([
            SqlRolePermission(
                role_id=role.id,
                resource_type=resource_type,
                resource_pattern=resource_id,
                permission=READ.name,
            )
            for resource_id in [visible, denied, *extra_ids]
        ])
    scanned = []
    original_fold = auth.fold_grants_for_key

    def count_rows(rows, key):
        scanned.append(len(rows))
        return original_fold(rows, key)

    fold = Mock(side_effect=count_rows)
    predicate = Mock(wraps=auth._role_based_read_predicate)
    search = Mock(wraps=getattr(tracking, search_method))
    monkeypatch.setattr(auth, "fold_grants_for_key", fold)
    monkeypatch.setattr(auth, "_role_based_read_predicate", predicate)
    monkeypatch.setattr(tracking, search_method, search)
    response = (
        client.post(path, **options)
        if collection == "logged-models"
        else client.get(path, **options)
    )
    assert response.status_code == 200
    if collection == "servers":
        assert [row["name"] for row in response.json()["mcp_servers"]] == [visible]
        predicate.assert_called_once_with(username, resource_type)
    elif collection == "logged-models":
        assert [row["info"]["experiment_id"] for row in response.json["models"]] == [visible]
        assert search.call_args.kwargs["experiment_ids"] == [visible]
        predicate.assert_called()
    else:
        assert [row["experiment_id"] for row in response.json["experiments"]] == [visible]
        predicate.assert_called_once_with(username, resource_type)
    fold.assert_called()
    search.assert_called_once()
    assert sum(scanned) < 12 * grant_count
