from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from flask import Flask
from starlette.testclient import TestClient
from werkzeug.datastructures import Authorization

from mlflow.entities.mcp_server import MCPStatus
from mlflow.server import auth, handlers
from mlflow.server.auth.permissions import DENY, READ
from mlflow.store.entities.paged_list import PagedList
from mlflow.utils import workspace_context

from tests.server.auth.test_auth_search_scope import (
    large_scope_search_setup as large_scope_search_setup,
)
from tests.server.test_mcp_server_api import _create_registry_fastapi_app


@pytest.fixture
def search_client(large_scope_search_setup, monkeypatch, collection, prefix):
    auth_store, _, _, _, username, password = large_scope_search_setup
    if collection in {"experiments", "registered-models"}:
        handler = (
            handlers._search_experiments
            if collection == "experiments"
            else handlers._search_registered_models
        )
        app = Flask(__name__)
        app.before_request(auth._before_request)
        app.after_request(auth._after_request)
        path = f"{prefix}/2.0/mlflow/{collection}/search"
        app.add_url_rule(path, view_func=handler, methods=["GET"])
        client = app.test_client()
        headers = {
            "Authorization": Authorization(
                "basic", {"username": username, "password": password}
            ).to_header()
        }

        def get(params, expected_status=200):
            response = client.get(path, query_string=params, headers=headers)
            assert response.status_code == expected_status
            return response.json

    else:
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
        if collection == "endpoints":
            path += "/endpoints"

        def get(params, expected_status=200):
            response = client.get(path, params=params)
            assert response.status_code == expected_status
            return response.json()

    return get


def create_row(tracking, collection, label):
    if collection == "experiments":
        resource_id = tracking.create_experiment(label)
        return resource_id, tracking.get_experiment(resource_id)
    name = f"com.example/{label}"
    server = tracking.create_mcp_server(name=name)
    if collection == "servers":
        return name, server
    tracking.create_mcp_server_version({"name": name, "version": "1.0.0"}, status=MCPStatus.ACTIVE)
    tracking.set_mcp_server_alias(name, "live", "1.0.0")
    return name, tracking.create_mcp_access_endpoint(
        name, f"https://example.com/{label}", server_alias="live"
    )


COLLECTION_FIELDS = {
    "experiments": ("search_experiments", "experiments", "experiment_id", "experiment"),
    "servers": ("search_mcp_servers", "mcp_servers", "name", "mcp_server"),
    "endpoints": (
        "search_mcp_access_endpoints",
        "mcp_access_endpoints",
        "server_name",
        "mcp_server",
    ),
}


@pytest.mark.parametrize("collection", list(COLLECTION_FIELDS))
@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("grant", ["named", "wildcard-deny"])
@pytest.mark.parametrize("max_results", [1, 2])
def test_scoped_search_paginates_in_backend_without_additional_queries(
    large_scope_search_setup, search_client, monkeypatch, collection, prefix, grant, max_results
):
    auth_store, tracking, _, role, _, _ = large_scope_search_setup
    method, result_key, name_key, resource_type = COLLECTION_FIELDS[collection]
    labels = ["hidden-0", "visible-a", "hidden-1", "visible-b", "hidden-2", "visible-c", "hidden-3"]
    ids = {label: create_row(tracking, collection, label)[0] for label in labels}
    with workspace_context.WorkspaceContext("team-b"):
        outside, _ = create_row(tracking, collection, "visible-outside")
    auth_store.add_role_permission(role.id, resource_type, outside, READ.name)
    if grant == "wildcard-deny":
        auth_store.add_role_permission(role.id, resource_type, "*", READ.name)
    for label, resource_id in ids.items():
        if label.startswith("visible") and grant == "named":
            auth_store.add_role_permission(role.id, resource_type, resource_id, READ.name)
        elif label.startswith("hidden") and grant == "wildcard-deny":
            auth_store.add_role_permission(role.id, resource_type, resource_id, DENY.name)

    backend_pages = []
    original_search = getattr(tracking, method)

    def search_page(**kwargs):
        page = original_search(**kwargs)
        token = page.token.decode() if isinstance(page.token, bytes) else page.token
        backend_pages.append(([getattr(row, name_key) for row in page], token))
        return page

    search = Mock(side_effect=search_page)
    monkeypatch.setattr(tracking, method, search)
    caller_filter = (
        "name != 'Default'" if collection == "experiments" else f"{name_key} != 'excluded'"
    )
    params = {
        "filter" if collection == "experiments" else "filter_string": caller_filter,
        "max_results": max_results,
        "order_by": "created_at ASC" if collection == "endpoints" else "name ASC",
    }
    if collection == "endpoints":
        params["server_alias"] = "live"
    pages = []
    for _ in range(4):
        data = search_client(params)
        page = [row[name_key] for row in data.get(result_key, [])]
        token = data.get("next_page_token")
        pages.append(page)
        assert backend_pages[-1] == (page, token or None)
        assert search.call_count == len(pages)
        if not token:
            break
        params["page_token"] = token
    else:
        pytest.fail("Scoped pagination did not terminate")
    visible = [ids[label] for label in labels if label.startswith("visible")]
    assert pages == [visible[i : i + max_results] for i in range(0, len(visible), max_results)]
    search.assert_called()
    for call in search.call_args_list:
        assert caller_filter in call.kwargs["filter_string"]
        assert call.kwargs["max_results"] == max_results
        assert call.kwargs["order_by"] == [params["order_by"]]
        if collection == "endpoints":
            assert call.kwargs["server_alias"] == "live"
            assert call.kwargs["server_version"] is None


@pytest.mark.parametrize("collection", list(COLLECTION_FIELDS))
@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("empty", [True, False])
def test_scoped_search_preserves_opaque_backend_continuation(
    large_scope_search_setup, search_client, monkeypatch, collection, prefix, empty
):
    auth_store, tracking, _, role, _, _ = large_scope_search_setup
    method, result_key, name_key, resource_type = COLLECTION_FIELDS[collection]
    resource_id, row = create_row(tracking, collection, "visible")
    auth_store.add_role_permission(role.id, resource_type, resource_id, READ.name)
    token = "opaque-backend-continuation"
    search = Mock(return_value=PagedList([] if empty else [row], token))
    monkeypatch.setattr(tracking, method, search)
    data = search_client({"max_results": 2, "page_token": "opaque-request-continuation"})
    assert [item[name_key] for item in data.get(result_key, [])] == ([] if empty else [resource_id])
    assert data["next_page_token"] == token
    search.assert_called_once()
    assert search.call_args.kwargs["page_token"] == "opaque-request-continuation"
    assert search.call_args.kwargs["max_results"] == 2
    assert resource_id in search.call_args.kwargs["filter_string"]


@pytest.mark.parametrize("collection", ["registered-models"])
@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("empty_pages", [0, 2])
@pytest.mark.parametrize("grant", ["named", "wildcard-deny"])
def test_retained_registry_filter_fills_pages_with_opaque_tokens(
    large_scope_search_setup, search_client, monkeypatch, collection, prefix, empty_pages, grant
):
    auth_store, _, registry, role, _, _ = large_scope_search_setup
    names = ["hidden-0", "visible-a", "hidden-1", "visible-b", "hidden-2", "visible-c", "hidden-3"]
    rows = [registry.create_registered_model(name) for name in names]
    if grant == "wildcard-deny":
        auth_store.add_role_permission(role.id, "registered_model", "*", READ.name)
    for name in names:
        if name.startswith("visible") and grant == "named":
            auth_store.add_role_permission(role.id, "registered_model", name, READ.name)
        elif name.startswith("hidden") and grant == "wildcard-deny":
            auth_store.add_role_permission(role.id, "registered_model", name, DENY.name)
    cursors = {}

    def token_for(state):
        token = f"opaque-{state[0]}-{state[1]}"
        cursors[token] = state
        return token

    def search_page(**kwargs):
        offset, empty_left = (
            cursors[kwargs["page_token"]] if kwargs.get("page_token") else (0, empty_pages)
        )
        if empty_left:
            return PagedList([], token_for((offset, empty_left - 1)))
        page = rows[offset : offset + kwargs["max_results"]]
        next_offset = offset + len(page)
        return PagedList(page, token_for((next_offset, 0)) if next_offset < len(rows) else None)

    search = Mock(side_effect=search_page)
    monkeypatch.setattr(registry, "search_registered_models", search)
    params = {"filter": "name != 'excluded'", "max_results": 2, "order_by": "name ASC"}
    pages = []
    for _ in range(3):
        data = search_client(params)
        pages.append([row["name"] for row in data.get("registered_models", [])])
        if not (token := data.get("next_page_token")):
            break
        params["page_token"] = token
    else:
        pytest.fail("Retained registry pagination did not terminate")
    assert pages == [["visible-a", "visible-b"], ["visible-c"]]
    search.assert_called()
    for call in search.call_args_list:
        assert call.kwargs["filter_string"] == "name != 'excluded'"
        assert call.kwargs["order_by"] == ["name ASC"]
        assert 0 < call.kwargs["max_results"] <= 2
        assert call.kwargs["page_token"] is None or call.kwargs["page_token"] in cursors


@pytest.mark.parametrize("collection", ["registered-models"])
@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("grant", ["named", "wildcard-deny"])
def test_retained_registry_filter_rejects_repeated_backend_token(
    large_scope_search_setup, search_client, monkeypatch, collection, prefix, grant
):
    auth_store, _, registry, role, _, _ = large_scope_search_setup
    hidden = registry.create_registered_model("hidden")
    visible = registry.create_registered_model("visible")
    if grant == "wildcard-deny":
        auth_store.add_role_permission(role.id, "registered_model", "*", READ.name)
        auth_store.add_role_permission(role.id, "registered_model", hidden.name, DENY.name)
    else:
        auth_store.add_role_permission(role.id, "registered_model", visible.name, READ.name)
    token = "opaque-repeated"
    search = Mock(
        side_effect=[
            PagedList([hidden], token),
            PagedList([hidden], token),
        ]
    )
    monkeypatch.setattr(registry, "search_registered_models", search)
    data = search_client({"max_results": 1}, expected_status=500)
    assert data["message"] == "Search backend repeated a pagination token."
    assert search.call_count == 2
    assert search.call_args.kwargs["page_token"] == token


@pytest.fixture
def retained_search_setup(large_scope_search_setup, collection, prefix, grant):
    auth_store, _, registry, role, username, password = large_scope_search_setup
    visible = registry.create_registered_model("visible")
    hidden = registry.create_registered_model("hidden")
    backend = registry
    if collection == "model-versions":
        visible = registry.create_model_version("visible", source="/model", run_id=None)
        hidden = registry.create_model_version("hidden", source="/model", run_id=None)
        method = "search_model_versions"
        handler = handlers._search_model_versions
        result_key = "model_versions"
    else:
        method = "search_registered_models"
        handler = handlers._search_registered_models
        result_key = "registered_models"
    payload = {"max_results": 1, "filter": "name != 'excluded'", "order_by": ["name ASC"]}
    resource_type = "registered_model"
    visible_id = "visible"
    hidden_id = "hidden"
    http_method = "GET"
    if grant == "named":
        auth_store.add_role_permission(role.id, resource_type, visible_id, READ.name)
    else:
        auth_store.add_role_permission(role.id, resource_type, "*", READ.name)
        auth_store.add_role_permission(role.id, resource_type, hidden_id, DENY.name)
    app = Flask(__name__)
    app.before_request(auth._before_request)
    app.after_request(auth._after_request)
    path = f"{prefix}/2.0/mlflow/{collection}/search"
    app.add_url_rule(path, view_func=handler, methods=[http_method])
    client = app.test_client()
    headers = {
        "Authorization": Authorization(
            "basic", {"username": username, "password": password}
        ).to_header()
    }

    def search_request(page_token=None):
        payload["page_token"] = page_token
        options = {"json": payload} if http_method == "POST" else {"query_string": payload}
        return client.open(path, method=http_method, headers=headers, **options)

    return backend, method, visible, hidden, result_key, search_request


@pytest.mark.parametrize("collection", ["registered-models", "model-versions"])
@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("grant", ["named", "wildcard-deny"])
@pytest.mark.parametrize("cycle_length", [1, 2])
def test_retained_search_rejects_continuation_cycles(
    retained_search_setup, monkeypatch, cycle_length
):
    backend, method, _, hidden, _, search_request = retained_search_setup
    tokens = [f"opaque-private-{index}" for index in range(cycle_length)]
    search = Mock(
        side_effect=[
            PagedList([hidden], tokens[0]),
            *[
                PagedList([hidden], tokens[(index + 1) % cycle_length])
                for index in range(cycle_length)
            ],
        ]
    )
    monkeypatch.setattr(backend, method, search)
    response = search_request()
    assert response.status_code == 500
    assert response.json["error_code"] == "INTERNAL_ERROR"
    assert response.json["message"] == "Search backend repeated a pagination token."
    assert all(token not in response.get_data(as_text=True) for token in tokens)
    assert search.call_count == cycle_length + 1
    assert [call.kwargs["page_token"] for call in search.call_args_list] == [None, *tokens]


@pytest.mark.parametrize("collection", ["registered-models", "model-versions"])
@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("grant", ["named", "wildcard-deny"])
@pytest.mark.parametrize("readable_tail", [False, True])
def test_retained_search_preserves_progressing_opaque_continuations(
    retained_search_setup, monkeypatch, readable_tail
):
    backend, method, visible, hidden, result_key, search_request = retained_search_setup
    token = "opaque-first"
    search = Mock(
        side_effect=[
            PagedList([hidden], token),
            PagedList([], "opaque-empty"),
            PagedList([hidden], "opaque-hidden"),
            PagedList([visible] if readable_tail else [], None),
        ]
    )
    monkeypatch.setattr(backend, method, search)
    response = search_request()
    assert response.status_code == 200
    assert len(response.json.get(result_key, [])) == int(readable_tail)
    assert not response.json.get("next_page_token")
    assert search.call_count == 4
    assert [call.kwargs["page_token"] for call in search.call_args_list] == [
        None,
        token,
        "opaque-empty",
        "opaque-hidden",
    ]


@pytest.mark.parametrize("collection", ["registered-models", "model-versions"])
@pytest.mark.parametrize("prefix", ["/api", "/ajax-api"])
@pytest.mark.parametrize("grant", ["named", "wildcard-deny"])
@pytest.mark.parametrize("refill", [False, True])
def test_retained_search_stops_at_full_page_without_lookahead(
    retained_search_setup, monkeypatch, refill
):
    backend, method, visible, hidden, result_key, search_request = retained_search_setup
    token = "opaque-unreadable-tail"
    search = Mock(
        side_effect=[
            *([PagedList([hidden], "opaque-refill")] if refill else []),
            PagedList([visible], token),
            PagedList([hidden], None),
        ]
    )
    monkeypatch.setattr(backend, method, search)

    response = search_request()

    assert response.status_code == 200
    assert [row["name"] for row in response.json[result_key]] == [visible.name]
    assert response.json["next_page_token"] == token
    assert search.call_count == 1 + int(refill)
    assert [call.kwargs["page_token"] for call in search.call_args_list] == (
        [None, "opaque-refill"] if refill else [None]
    )

    terminal_response = search_request(token)

    assert terminal_response.status_code == 200
    assert terminal_response.json.get(result_key, []) == []
    assert not terminal_response.json.get("next_page_token")
    assert search.call_count == 2 + int(refill)
    assert search.call_args.kwargs["page_token"] == token
    for call in search.call_args_list:
        assert call.kwargs["max_results"] == 1
        assert call.kwargs["filter_string"] == "name != 'excluded'"
        assert call.kwargs["order_by"] == ["name ASC"]
