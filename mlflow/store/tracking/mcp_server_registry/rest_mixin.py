"""REST implementation of MCPServerRegistryMixin."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from mlflow.entities.mcp_access_endpoint import MCPAccessEndpoint
from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPServer, MCPStatus, MCPTool
from mlflow.entities.mcp_server_version import MCPServerVersion
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking.mcp_server_registry.abstract_mixin import NOT_SET, MCPIcon
from mlflow.utils.rest_utils import http_request, verify_rest_response

_MCP_API_PREFIX = "/api/3.0/mlflow/mcp-servers"


def _encode_path_param(value: str) -> str:
    return quote(str(value), safe="")


def _server_path(name: str) -> str:
    return f"/{_encode_path_param(name)}"


class RestMCPServerRegistryMixin:
    """REST implementation of MCPServerRegistryMixin.

    Expects the implementing class to provide ``get_host_creds()``.
    Uses ``http_request`` directly (no protobuf) since the MCP server
    registry endpoints use Pydantic/JSON.
    """

    def _mcp_request(self, method: str, path: str, json=None, params=None):
        self._validate_workspace_support_if_specified()
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
        created_by: str | None = None,
    ) -> MCPServer:
        body = {"name": name}
        if description is not None:
            body["description"] = description
        if icons is not None:
            body["icons"] = icons
        data = self._mcp_request("POST", "", json=body)
        return MCPServer.from_dict(data)

    def get_mcp_server(self, name: str) -> MCPServer:
        data = self._mcp_request("GET", _server_path(name))
        return MCPServer.from_dict(data)

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
            servers = [MCPServer.from_dict(s) for s in data["mcp_servers"]]
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
        last_updated_by: str | None = None,
    ) -> MCPServer:
        body: dict[str, Any] = {}
        if description is not NOT_SET:
            body["description"] = description
        if display_name is not NOT_SET:
            body["display_name"] = display_name
        if icons is not NOT_SET:
            body["icons"] = icons
        data = self._mcp_request("PATCH", _server_path(name), json=body)
        return MCPServer.from_dict(data)

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
        created_by: str | None = None,
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
        return MCPServerVersion.from_dict(data)

    def get_mcp_server_version(self, name: str, version: str) -> MCPServerVersion:
        data = self._mcp_request(
            "GET", f"{_server_path(name)}/versions/{_encode_path_param(version)}"
        )
        return MCPServerVersion.from_dict(data)

    def get_mcp_server_version_by_alias(self, name: str, alias: str) -> MCPServerVersion:
        data = self._mcp_request("GET", f"{_server_path(name)}/aliases/{_encode_path_param(alias)}")
        return MCPServerVersion.from_dict(data)

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
        try:
            if not isinstance(data, dict):
                raise MlflowException.invalid_parameter_value(
                    "Failed to parse search response: expected a dictionary"
                )
            if data.get("mcp_server_versions") is None:
                raise MlflowException.invalid_parameter_value(
                    "Failed to parse search response: mcp_server_versions field is null"
                )
            versions = [MCPServerVersion.from_dict(v) for v in data["mcp_server_versions"]]
            return PagedList(versions, data.get("next_page_token"))
        except (KeyError, TypeError, ValueError) as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse search response: {e}"
            ) from None

    def update_mcp_server_version(
        self,
        name: str,
        version: str,
        display_name: str | None = NOT_SET,
        status: MCPStatus | None = NOT_SET,
        tools: list[MCPTool] | None = NOT_SET,
        last_updated_by: str | None = None,
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
        return MCPServerVersion.from_dict(data)

    def delete_mcp_server_version(self, name: str, version: str) -> None:
        self._mcp_request("DELETE", f"{_server_path(name)}/versions/{_encode_path_param(version)}")

    # --- MCPAccessEndpoint operations ---

    def create_mcp_access_endpoint(
        self,
        server_name: str,
        url: str,
        transport_type: MCPRemoteTransportType = MCPRemoteTransportType.STREAMABLE_HTTP,
        server_version: str | None = None,
        server_alias: str | None = None,
        created_by: str | None = None,
    ) -> MCPAccessEndpoint:
        body: dict[str, Any] = {
            "url": url,
            "transport_type": str(transport_type),
        }
        if server_version is not None:
            body["server_version"] = server_version
        if server_alias is not None:
            body["server_alias"] = server_alias
        data = self._mcp_request("POST", f"{_server_path(server_name)}/endpoints", json=body)
        return MCPAccessEndpoint.from_dict(data)

    def get_mcp_access_endpoint(self, server_name: str, endpoint_id: str) -> MCPAccessEndpoint:
        data = self._mcp_request(
            "GET", f"{_server_path(server_name)}/endpoints/{_encode_path_param(endpoint_id)}"
        )
        return MCPAccessEndpoint.from_dict(data)

    def search_mcp_access_endpoints(
        self,
        server_name: str | None = None,
        server_version: str | None = None,
        server_alias: str | None = None,
        filter_string: str | None = None,
        max_results: int = SEARCH_MAX_RESULTS_DEFAULT,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[MCPAccessEndpoint]:
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
        if server_name is None:
            path = "/endpoints"
        elif isinstance(server_name, str) and server_name.strip():
            path = f"{_server_path(server_name)}/endpoints"
        else:
            raise MlflowException.invalid_parameter_value(
                "server_name must be a non-empty string when provided"
            )
        data = self._mcp_request("GET", path, params=params)
        try:
            if not isinstance(data, dict):
                raise MlflowException.invalid_parameter_value(
                    "Failed to parse search response: expected a dictionary"
                )
            if data.get("mcp_access_endpoints") is None:
                raise MlflowException.invalid_parameter_value(
                    "Failed to parse search response: mcp_access_endpoints field is null"
                )
            endpoints = [MCPAccessEndpoint.from_dict(e) for e in data["mcp_access_endpoints"]]
            return PagedList(endpoints, data.get("next_page_token"))
        except (KeyError, TypeError, ValueError) as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse search response: {e}"
            ) from None

    def update_mcp_access_endpoint(
        self,
        server_name: str,
        endpoint_id: str,
        server_version: str | None = NOT_SET,
        server_alias: str | None = NOT_SET,
        url: str | None = NOT_SET,
        transport_type: MCPRemoteTransportType | None = NOT_SET,
        last_updated_by: str | None = None,
    ) -> MCPAccessEndpoint:
        body: dict[str, Any] = {}
        if server_version is not NOT_SET:
            body["server_version"] = server_version
        if server_alias is not NOT_SET:
            body["server_alias"] = server_alias
        if url is not NOT_SET:
            body["url"] = url
        if transport_type is not NOT_SET:
            body["transport_type"] = str(transport_type) if transport_type is not None else None
        data = self._mcp_request(
            "PATCH",
            f"{_server_path(server_name)}/endpoints/{_encode_path_param(endpoint_id)}",
            json=body,
        )
        return MCPAccessEndpoint.from_dict(data)

    def delete_mcp_access_endpoint(self, server_name: str, endpoint_id: str) -> None:
        self._mcp_request(
            "DELETE", f"{_server_path(server_name)}/endpoints/{_encode_path_param(endpoint_id)}"
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
        raise NotImplementedError(self.__class__.__name__)

    def get_mcp_server_versions_for_trace(
        self,
        trace_id: str,
    ) -> list[MCPServerVersion]:
        raise NotImplementedError(self.__class__.__name__)
