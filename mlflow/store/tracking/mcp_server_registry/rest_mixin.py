"""REST implementation of MCPServerRegistryMixin.

Makes HTTP calls to the FastAPI endpoints defined in mlflow.server.mcp_server_api.
Follows the direct-HTTP pattern used by online scoring config methods in rest_store.py.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from mlflow.entities.mcp_access_binding import MCPAccessBinding
from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPServer, MCPStatus, MCPTool
from mlflow.entities.mcp_server_version import MCPServerVersion
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking.mcp_server_registry.abstract_mixin import NOT_SET, MCPIcon
from mlflow.utils.rest_utils import http_request, verify_rest_response

_MCP_API_PREFIX = "/ajax-api/3.0/mlflow/mcp-servers"


def _encode_path_param(value: str | int) -> str:
    return quote(str(value), safe="")


def _server_path(name: str) -> str:
    return f"/{_encode_path_param(name)}"


def _server_from_response(data: dict[str, Any]) -> MCPServer:
    try:
        if not isinstance(data, dict):
            raise MlflowException.invalid_parameter_value(
                "Failed to parse MCP server response: expected a dictionary"
            )
        return MCPServer(
            name=data["name"],
            display_name=data.get("display_name"),
            description=data.get("description"),
            icons=data.get("icons"),
            workspace=data.get("workspace"),
            status=MCPStatus(data["status"]) if data.get("status") else None,
            tags=data.get("tags", {}),
            aliases={a["alias"]: a["version"] for a in data.get("aliases", [])},
            access_bindings=[
                _binding_summary_from_response(b) for b in data.get("access_bindings", [])
            ],
            latest_version=data.get("latest_version"),
            created_by=data.get("created_by"),
            last_updated_by=data.get("last_updated_by"),
            creation_timestamp=data.get("creation_timestamp"),
            last_updated_timestamp=data.get("last_updated_timestamp"),
        )
    except KeyError as e:
        raise MlflowException.invalid_parameter_value(
            f"Failed to parse MCP server response: missing required field {e}"
        ) from None
    except (ValueError, TypeError) as e:
        raise MlflowException.invalid_parameter_value(
            f"Failed to parse MCP server response: {e}"
        ) from None


def _version_from_response(data: dict[str, Any]) -> MCPServerVersion:
    try:
        if not isinstance(data, dict):
            raise MlflowException.invalid_parameter_value(
                "Failed to parse MCP server version response: expected a dictionary"
            )
        tools = None
        if "tools" in data and data["tools"] is not None:
            try:
                tools = [MCPTool.from_dict(t) for t in data["tools"]]
            except MlflowException as e:
                raise MlflowException.invalid_parameter_value(
                    f"Failed to parse MCP server version response: {e.message}"
                ) from None
        return MCPServerVersion(
            name=data["name"],
            version=data["version"],
            server_json=data["server_json"],
            display_name=data.get("display_name"),
            workspace=data.get("workspace"),
            status=MCPStatus(data["status"]) if data.get("status") else MCPStatus.DRAFT,
            tools=tools,
            aliases=data.get("aliases", []),
            tags=data.get("tags", {}),
            source=data.get("source"),
            created_by=data.get("created_by"),
            last_updated_by=data.get("last_updated_by"),
            creation_timestamp=data.get("creation_timestamp"),
            last_updated_timestamp=data.get("last_updated_timestamp"),
        )
    except KeyError as e:
        raise MlflowException.invalid_parameter_value(
            f"Failed to parse MCP server version response: missing required field {e}"
        ) from None
    except (ValueError, TypeError) as e:
        raise MlflowException.invalid_parameter_value(
            f"Failed to parse MCP server version response: {e}"
        ) from None


def _binding_from_response(data: dict[str, Any]) -> MCPAccessBinding:
    try:
        if not isinstance(data, dict):
            raise MlflowException.invalid_parameter_value(
                "Failed to parse MCP access binding response: expected a dictionary"
            )
        return MCPAccessBinding(
            binding_id=data["binding_id"],
            server_name=data["server_name"],
            endpoint_url=data["endpoint_url"],
            transport_type=MCPRemoteTransportType(data.get("transport_type", "streamable-http")),
            workspace=data.get("workspace"),
            server_version=data.get("server_version"),
            server_alias=data.get("server_alias"),
            resolved_version=(
                None
                if data.get("resolved_version") is None
                else _version_from_response(data["resolved_version"])
            ),
            created_by=data.get("created_by"),
            last_updated_by=data.get("last_updated_by"),
            creation_timestamp=data.get("creation_timestamp"),
            last_updated_timestamp=data.get("last_updated_timestamp"),
        )
    except KeyError as e:
        raise MlflowException.invalid_parameter_value(
            f"Failed to parse MCP access binding response: missing required field {e}"
        ) from None
    except (ValueError, TypeError) as e:
        raise MlflowException.invalid_parameter_value(
            f"Failed to parse MCP access binding response: {e}"
        ) from None


def _binding_summary_from_response(data: dict[str, Any]) -> MCPAccessBinding:
    try:
        if not isinstance(data, dict):
            raise MlflowException.invalid_parameter_value(
                "Failed to parse MCP access binding summary: expected a dictionary"
            )
        return MCPAccessBinding(
            binding_id=data["binding_id"],
            server_name=data.get("server_name", ""),
            endpoint_url=data["endpoint_url"],
            transport_type=MCPRemoteTransportType(data.get("transport_type", "streamable-http")),
            workspace=data.get("workspace"),
            server_version=data.get("server_version"),
            server_alias=data.get("server_alias"),
            resolved_version=(
                None
                if data.get("resolved_version") is None
                else _version_from_response(data["resolved_version"])
            ),
            created_by=data.get("created_by"),
            last_updated_by=data.get("last_updated_by"),
            creation_timestamp=data.get("creation_timestamp"),
            last_updated_timestamp=data.get("last_updated_timestamp"),
        )
    except KeyError as e:
        raise MlflowException.invalid_parameter_value(
            f"Failed to parse MCP access binding summary: missing required field {e}"
        ) from None
    except (ValueError, TypeError) as e:
        raise MlflowException.invalid_parameter_value(
            f"Failed to parse MCP access binding summary: {e}"
        ) from None


class RestMCPServerRegistryMixin:
    """REST implementation of MCPServerRegistryMixin.

    Expects the implementing class to provide ``get_host_creds()``.
    Uses ``http_request`` directly (no protobuf) since the MCP server
    registry endpoints use Pydantic/JSON.
    """

    def _mcp_request(self, method: str, path: str, json=None, params=None):
        endpoint = f"{_MCP_API_PREFIX}{path}"
        response = http_request(
            host_creds=self.get_host_creds(),
            endpoint=endpoint,
            method=method,
            json=json,
            params=params,
        )
        verify_rest_response(response, endpoint)
        if not response.text:
            return None
        return response.json()

    # --- MCPServer operations ---

    def create_mcp_server(
        self,
        name: str,
        description: str | None = None,
        icons: list[MCPIcon] | None = None,
    ) -> MCPServer:
        body = {"name": name}
        if description is not None:
            body["description"] = description
        if icons is not None:
            body["icons"] = icons
        data = self._mcp_request("POST", "", json=body)
        return _server_from_response(data)

    def get_mcp_server(self, name: str) -> MCPServer:
        data = self._mcp_request("GET", _server_path(name))
        return _server_from_response(data)

    def search_mcp_servers(
        self,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[MCPServer]:
        params: dict[str, Any] = {"max_results": max_results}
        if filter_string is not None:
            params["filter_string"] = filter_string
        if order_by is not None:
            params["order_by"] = order_by
        if page_token is not None:
            params["page_token"] = page_token
        data = self._mcp_request("GET", "", params=params)
        try:
            if not isinstance(data, dict):
                raise MlflowException.invalid_parameter_value(
                    "Failed to parse search response: expected a dictionary"
                )
            if data.get("mcp_servers") is None:
                raise MlflowException.invalid_parameter_value(
                    "Failed to parse search response: mcp_servers field is null"
                )
            servers = [_server_from_response(s) for s in data["mcp_servers"]]
            return PagedList(servers, data.get("next_page_token"))
        except (KeyError, TypeError, ValueError) as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse search response: {e}"
            ) from None

    def update_mcp_server(
        self,
        name: str,
        description: str | None = NOT_SET,
        display_name: str | None = NOT_SET,
        icons: list[MCPIcon] | None = NOT_SET,
        latest_version: str | None = NOT_SET,
    ) -> MCPServer:
        body: dict[str, Any] = {}
        if description is not NOT_SET:
            body["description"] = description
        if display_name is not NOT_SET:
            body["display_name"] = display_name
        if icons is not NOT_SET:
            body["icons"] = icons
        if latest_version is not NOT_SET:
            body["latest_version"] = latest_version
        data = self._mcp_request("PATCH", _server_path(name), json=body)
        return _server_from_response(data)

    def delete_mcp_server(self, name: str) -> None:
        self._mcp_request("DELETE", _server_path(name))

    # --- MCPServerVersion operations ---

    def create_mcp_server_version(
        self,
        server_json: dict[str, Any],
        display_name: str | None = None,
        source: str | None = None,
        status: MCPStatus | None = None,
        tools: list[MCPTool] | None = None,
    ) -> MCPServerVersion:
        name = server_json.get("name")
        version = server_json.get("version")
        if not name or not version:
            raise MlflowException.invalid_parameter_value(
                "server_json must contain 'name' and 'version' keys"
            )
        body: dict[str, Any] = {"server_json": server_json}
        if display_name is not None:
            body["display_name"] = display_name
        if source is not None:
            body["source"] = source
        if status is not None:
            body["status"] = str(status)
        if tools is not None:
            body["tools"] = [t.to_dict() for t in tools]
        data = self._mcp_request("POST", f"{_server_path(name)}/versions", json=body)
        return _version_from_response(data)

    def get_mcp_server_version(self, name: str, version: str) -> MCPServerVersion:
        data = self._mcp_request(
            "GET", f"{_server_path(name)}/versions/{_encode_path_param(version)}"
        )
        return _version_from_response(data)

    def get_mcp_server_version_by_alias(self, name: str, alias: str) -> MCPServerVersion:
        data = self._mcp_request("GET", f"{_server_path(name)}/aliases/{_encode_path_param(alias)}")
        return _version_from_response(data)

    def get_latest_mcp_server_version(self, name: str) -> MCPServerVersion:
        return self.get_mcp_server_version_by_alias(name, "latest")

    def search_mcp_server_versions(
        self,
        name: str,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[MCPServerVersion]:
        params: dict[str, Any] = {"max_results": max_results}
        if filter_string is not None:
            params["filter_string"] = filter_string
        if order_by is not None:
            params["order_by"] = order_by
        if page_token is not None:
            params["page_token"] = page_token
        data = self._mcp_request("GET", f"{_server_path(name)}/versions", params=params)
        versions = [_version_from_response(v) for v in data["mcp_server_versions"]]
        return PagedList(versions, data.get("next_page_token"))

    def update_mcp_server_version(
        self,
        name: str,
        version: str,
        display_name: str | None = NOT_SET,
        status: MCPStatus | None = NOT_SET,
        tools: list[MCPTool] | None = NOT_SET,
    ) -> MCPServerVersion:
        body: dict[str, Any] = {}
        if display_name is not NOT_SET:
            body["display_name"] = display_name
        if status is not NOT_SET:
            body["status"] = str(status) if status is not None else None
        if tools is not NOT_SET:
            body["tools"] = None if tools is None else [t.to_dict() for t in tools]
        data = self._mcp_request(
            "PATCH", f"{_server_path(name)}/versions/{_encode_path_param(version)}", json=body
        )
        return _version_from_response(data)

    def delete_mcp_server_version(self, name: str, version: str) -> None:
        self._mcp_request("DELETE", f"{_server_path(name)}/versions/{_encode_path_param(version)}")

    # --- MCPAccessBinding operations ---

    def create_mcp_access_binding(
        self,
        server_name: str,
        endpoint_url: str,
        transport_type: MCPRemoteTransportType = MCPRemoteTransportType.STREAMABLE_HTTP,
        server_version: str | None = None,
        server_alias: str | None = None,
    ) -> MCPAccessBinding:
        body: dict[str, Any] = {
            "endpoint_url": endpoint_url,
            "transport_type": str(transport_type),
        }
        if server_version is not None:
            body["server_version"] = server_version
        if server_alias is not None:
            body["server_alias"] = server_alias
        data = self._mcp_request("POST", f"{_server_path(server_name)}/bindings", json=body)
        return _binding_from_response(data)

    def get_mcp_access_binding(self, server_name: str, binding_id: int) -> MCPAccessBinding:
        data = self._mcp_request(
            "GET", f"{_server_path(server_name)}/bindings/{_encode_path_param(binding_id)}"
        )
        return _binding_from_response(data)

    def search_mcp_access_bindings(
        self,
        server_name: str | None = None,
        server_version: str | None = None,
        server_alias: str | None = None,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[MCPAccessBinding]:
        params: dict[str, Any] = {"max_results": max_results}
        if filter_string is not None:
            params["filter_string"] = filter_string
        if order_by is not None:
            params["order_by"] = order_by
        if page_token is not None:
            params["page_token"] = page_token
        if server_version is not None:
            params["server_version"] = server_version
        if server_alias is not None:
            params["server_alias"] = server_alias
        path = f"{_server_path(server_name)}/bindings" if server_name else "/bindings"
        data = self._mcp_request("GET", path, params=params)
        bindings = [_binding_from_response(b) for b in data["mcp_access_bindings"]]
        return PagedList(bindings, data.get("next_page_token"))

    def update_mcp_access_binding(
        self,
        server_name: str,
        binding_id: int,
        server_version: str | None = NOT_SET,
        server_alias: str | None = NOT_SET,
        endpoint_url: str | None = NOT_SET,
        transport_type: MCPRemoteTransportType | None = NOT_SET,
    ) -> MCPAccessBinding:
        body: dict[str, Any] = {}
        if server_version is not NOT_SET:
            body["server_version"] = server_version
        if server_alias is not NOT_SET:
            body["server_alias"] = server_alias
        if endpoint_url is not NOT_SET:
            body["endpoint_url"] = endpoint_url
        if transport_type is not NOT_SET:
            body["transport_type"] = str(transport_type) if transport_type is not None else None
        data = self._mcp_request(
            "PATCH",
            f"{_server_path(server_name)}/bindings/{_encode_path_param(binding_id)}",
            json=body,
        )
        return _binding_from_response(data)

    def delete_mcp_access_binding(self, server_name: str, binding_id: int) -> None:
        self._mcp_request(
            "DELETE", f"{_server_path(server_name)}/bindings/{_encode_path_param(binding_id)}"
        )

    # --- Tag operations ---

    def set_mcp_server_tag(self, name: str, key: str, value: str) -> None:
        self._mcp_request("POST", f"{_server_path(name)}/tags", json={"key": key, "value": value})

    def delete_mcp_server_tag(self, name: str, key: str) -> None:
        self._mcp_request("DELETE", f"{_server_path(name)}/tags/{_encode_path_param(key)}")

    def set_mcp_server_version_tag(self, name: str, version: str, key: str, value: str) -> None:
        self._mcp_request(
            "POST",
            f"{_server_path(name)}/versions/{_encode_path_param(version)}/tags",
            json={"key": key, "value": value},
        )

    def delete_mcp_server_version_tag(self, name: str, version: str, key: str) -> None:
        self._mcp_request(
            "DELETE",
            f"{_server_path(name)}/versions/{_encode_path_param(version)}/tags/"
            f"{_encode_path_param(key)}",
        )

    # --- Alias operations ---

    def set_mcp_server_alias(self, name: str, alias: str, version: str) -> None:
        self._mcp_request(
            "POST", f"{_server_path(name)}/aliases", json={"alias": alias, "version": version}
        )

    def delete_mcp_server_alias(self, name: str, alias: str) -> None:
        self._mcp_request("DELETE", f"{_server_path(name)}/aliases/{_encode_path_param(alias)}")

    # --- Trace linking ---

    def link_mcp_server_versions_to_trace(
        self,
        trace_id: str,
        mcp_servers: list[MCPServerVersion],
    ) -> None:
        self._mcp_request(
            "POST",
            "/traces/link-versions",
            json={
                "trace_id": trace_id,
                "mcp_server_versions": [
                    {"name": sv.name, "version": sv.version} for sv in mcp_servers
                ],
            },
        )

    def get_mcp_server_versions_for_trace(
        self,
        trace_id: str,
    ) -> list[MCPServerVersion]:
        data = self._mcp_request(
            "GET",
            "/traces/mcp-server-versions",
            params={"trace_id": trace_id},
        )
        return [_version_from_response(v) for v in data["mcp_server_versions"]]
