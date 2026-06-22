import json
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.rest_store import RestStore
from mlflow.utils.rest_utils import MlflowHostCreds


@pytest.fixture
def rest_store():
    creds = MlflowHostCreds(host="http://test-server")
    return RestStore(lambda: creds)


@pytest.fixture
def mock_http_request():
    with mock.patch(
        "mlflow.store.tracking.mcp_server_registry.rest_mixin.http_request"
    ) as mock_req:
        yield mock_req


def _make_response(status_code=200, text="", json_data=None):
    response = mock.Mock()
    response.status_code = status_code
    response.text = text if json_data is None else json.dumps(json_data)
    response.json.return_value = json_data
    return response


def test_create_mcp_server_preserves_workspace(rest_store, mock_http_request):
    server_workspace = "production"
    mock_http_request.return_value = _make_response(
        json_data={
            "name": "com.example/test-server",
            "workspace": server_workspace,
            "status": "active",
            "created_by": "test-user",
            "creation_timestamp": 1234567890,
        }
    )

    server = rest_store.create_mcp_server(
        name="com.example/test-server",
        description="Test server",
    )

    assert server.workspace == server_workspace, (
        f"Expected workspace='{server_workspace}', "
        f"but got workspace='{server.workspace}' (likely client default)"
    )


def test_get_mcp_server_preserves_workspace(rest_store, mock_http_request):
    server_workspace = "staging"
    mock_http_request.return_value = _make_response(
        json_data={
            "name": "com.example/test-server",
            "workspace": server_workspace,
            "status": "active",
        }
    )

    server = rest_store.get_mcp_server("com.example/test-server")
    assert server.workspace == server_workspace


def test_create_mcp_server_version_preserves_workspace(rest_store, mock_http_request):
    server_workspace = "development"
    mock_http_request.return_value = _make_response(
        json_data={
            "name": "com.example/test-server",
            "version": "1.0.0",
            "server_json": {"name": "com.example/test-server", "version": "1.0.0"},
            "workspace": server_workspace,
            "status": "draft",
        }
    )

    version = rest_store.create_mcp_server_version(
        server_json={"name": "com.example/test-server", "version": "1.0.0"}
    )
    assert version.workspace == server_workspace


def test_get_mcp_server_version_preserves_workspace(rest_store, mock_http_request):
    server_workspace = "production"
    mock_http_request.return_value = _make_response(
        json_data={
            "name": "com.example/test-server",
            "version": "1.0.0",
            "server_json": {"name": "com.example/test-server", "version": "1.0.0"},
            "workspace": server_workspace,
            "status": "active",
        }
    )

    version = rest_store.get_mcp_server_version("com.example/test-server", "1.0.0")
    assert version.workspace == server_workspace


def test_create_mcp_access_binding_preserves_workspace(rest_store, mock_http_request):
    server_workspace = "production"
    mock_http_request.return_value = _make_response(
        json_data={
            "binding_id": 1,
            "server_name": "com.example/test-server",
            "endpoint_url": "http://example.com",
            "transport_type": "streamable-http",
            "workspace": server_workspace,
            "created_by": "test-user",
            "creation_timestamp": 1234567890,
        }
    )

    binding = rest_store.create_mcp_access_binding(
        server_name="com.example/test-server",
        endpoint_url="http://example.com",
    )
    assert binding.workspace == server_workspace


def test_get_mcp_access_binding_preserves_workspace(rest_store, mock_http_request):
    server_workspace = "production"
    mock_http_request.return_value = _make_response(
        json_data={
            "binding_id": 1,
            "server_name": "com.example/test-server",
            "endpoint_url": "http://example.com",
            "transport_type": "streamable-http",
            "workspace": server_workspace,
        }
    )

    binding = rest_store.get_mcp_access_binding("com.example/test-server", 1)
    assert binding.workspace == server_workspace


def test_get_mcp_server_preserves_binding_metadata(rest_store, mock_http_request):
    mock_http_request.return_value = _make_response(
        json_data={
            "name": "com.example/test-server",
            "status": "active",
            "access_bindings": [
                {
                    "binding_id": 1,
                    "server_name": "com.example/test-server",
                    "endpoint_url": "http://example.com",
                    "transport_type": "streamable-http",
                    "workspace": "production",
                    "created_by": "admin",
                    "last_updated_by": "admin",
                    "creation_timestamp": 1234567890,
                    "last_updated_timestamp": 1234567890,
                }
            ],
        }
    )

    server = rest_store.get_mcp_server("com.example/test-server")
    assert len(server.access_bindings) == 1

    binding = server.access_bindings[0]
    assert binding.workspace == "production"
    assert binding.created_by == "admin"
    assert binding.last_updated_by == "admin"
    assert binding.creation_timestamp == 1234567890
    assert binding.last_updated_timestamp == 1234567890


def test_search_mcp_servers_preserves_binding_metadata(rest_store, mock_http_request):
    mock_http_request.return_value = _make_response(
        json_data={
            "mcp_servers": [
                {
                    "name": "com.example/server1",
                    "status": "active",
                    "access_bindings": [
                        {
                            "binding_id": 1,
                            "server_name": "com.example/server1",
                            "endpoint_url": "http://example1.com",
                            "transport_type": "streamable-http",
                            "workspace": "production",
                            "created_by": "user1",
                            "last_updated_by": "user1",
                            "creation_timestamp": 1111111111,
                            "last_updated_timestamp": 1111111111,
                        }
                    ],
                }
            ],
            "next_page_token": None,
        }
    )

    results = rest_store.search_mcp_servers()
    assert len(results) == 1

    server = results[0]
    assert len(server.access_bindings) == 1

    binding = server.access_bindings[0]
    assert binding.workspace == "production"
    assert binding.created_by == "user1"
    assert binding.last_updated_by == "user1"
    assert binding.creation_timestamp == 1111111111
    assert binding.last_updated_timestamp == 1111111111


def test_missing_required_field_raises_mlflow_exception(rest_store, mock_http_request):
    mock_http_request.return_value = _make_response(
        json_data={
            "status": "active",
        }
    )

    with pytest.raises(MlflowException, match="Failed to parse.*name"):
        rest_store.get_mcp_server("com.example/test-server")


def test_invalid_status_enum_raises_mlflow_exception(rest_store, mock_http_request):
    mock_http_request.return_value = _make_response(
        json_data={
            "name": "com.example/test-server",
            "status": "invalid-status",
        }
    )

    with pytest.raises(MlflowException, match="Failed to parse.*status"):
        rest_store.get_mcp_server("com.example/test-server")


def test_invalid_transport_type_raises_mlflow_exception(rest_store, mock_http_request):
    mock_http_request.return_value = _make_response(
        json_data={
            "binding_id": 1,
            "server_name": "com.example/test-server",
            "endpoint_url": "http://example.com",
            "transport_type": "invalid-transport",
        }
    )

    with pytest.raises(MlflowException, match="Failed to parse.*transport"):
        rest_store.get_mcp_access_binding("com.example/test-server", 1)


def test_non_dict_response_raises_mlflow_exception(rest_store, mock_http_request):
    mock_http_request.return_value = _make_response(json_data="not a dict")

    with pytest.raises(MlflowException, match="Failed to parse"):
        rest_store.get_mcp_server("com.example/test-server")


def test_malformed_nested_tool_raises_mlflow_exception(rest_store, mock_http_request):
    mock_http_request.return_value = _make_response(
        json_data={
            "name": "com.example/test-server",
            "version": "1.0.0",
            "server_json": {"name": "com.example/test-server", "version": "1.0.0"},
            "status": "active",
            "tools": [{"invalid": "tool"}],
        }
    )

    with pytest.raises(MlflowException, match="Failed to parse"):
        rest_store.get_mcp_server_version("com.example/test-server", "1.0.0")


def test_null_where_object_expected_raises_mlflow_exception(rest_store, mock_http_request):
    mock_http_request.return_value = _make_response(
        json_data={
            "mcp_servers": None,
            "next_page_token": None,
        }
    )

    with pytest.raises(MlflowException, match="Failed to parse"):
        rest_store.search_mcp_servers()
