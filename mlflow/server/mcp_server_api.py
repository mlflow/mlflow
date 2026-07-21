from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from mlflow.entities.mcp_access_endpoint import MCPAccessEndpoint
from mlflow.entities.mcp_server import (
    MCPRemoteTransportType,
    MCPServer,
    MCPStatus,
    MCPTool,
    validate_mcp_server_name,
)
from mlflow.entities.mcp_server_version import MCPServerVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED, RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.utils.validation import (
    _MAX_MCP_ICONS_PER_LIST,
    _MAX_MCP_TOOLS_PER_LIST,
    _validate_mcp_icon_mime_type,
    _validate_mcp_icon_url,
)

if TYPE_CHECKING:
    from mlflow.store.tracking.mcp_server_registry.abstract_mixin import MCPIcon

_MCP_SERVER_AJAX_API_PREFIX = "/ajax-api/3.0/mlflow/mcp-servers"
_MCP_SERVER_API_PREFIX = "/api/3.0/mlflow/mcp-servers"


def get_mcp_server_api_route_prefixes() -> tuple[str, ...]:
    from mlflow.server.handlers import _add_static_prefix

    return (
        _add_static_prefix(_MCP_SERVER_AJAX_API_PREFIX),
        _add_static_prefix(_MCP_SERVER_API_PREFIX),
    )


def is_mcp_server_api_path(path: str) -> bool:
    return any(
        path == prefix or path.startswith(f"{prefix}/")
        for prefix in get_mcp_server_api_route_prefixes()
    )


class _BaseMCPIconPayload(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    src: str
    sizes: list[str] | None = None
    mimeType: str | None = None
    theme: str | None = None

    @model_serializer(mode="plain")
    def serialize(self) -> dict[str, Any]:
        icon = dict(self.model_extra or {})
        icon["src"] = self.src
        if self.sizes is not None:
            icon["sizes"] = self.sizes
        if self.mimeType is not None:
            icon["mimeType"] = self.mimeType
        if self.theme is not None:
            icon["theme"] = self.theme
        return icon


class MCPIconRequestPayload(_BaseMCPIconPayload):
    @field_validator("src")
    @classmethod
    def _validate_src(cls, value: str) -> str:
        _validate_mcp_icon_url(value)
        return value

    @field_validator("mimeType")
    @classmethod
    def _validate_mime_type(cls, value: str | None) -> str | None:
        _validate_mcp_icon_mime_type(value)
        return None if value is None else value.strip().lower()


class MCPIconResponsePayload(_BaseMCPIconPayload):
    pass


class ServerJSONRepositoryPayload(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    url: str
    source: str
    id: str | None = None
    subfolder: str | None = None


class ServerJSONPayload(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str
    version: str
    title: str | None = None
    description: str | None = None
    icons: list[MCPIconRequestPayload] | None = Field(
        default=None, max_length=_MAX_MCP_ICONS_PER_LIST
    )
    packages: list[ServerJSONPackagePayload] | None = None
    remotes: list[ServerJSONRemotePayload] | None = None
    repository: ServerJSONRepositoryPayload | None = None
    websiteUrl: str | None = None
    meta: dict[str, Any] | None = Field(None, alias="_meta")


class ServerJSONEnvironmentVariablePayload(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str
    description: str | None = None
    isRequired: bool | None = None
    isSecret: bool | None = None


class ServerJSONPackagePayload(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    registryType: str
    identifier: str
    transport: Any
    registryBaseUrl: str | None = None
    version: str | None = None
    environmentVariables: list[ServerJSONEnvironmentVariablePayload] | None = None


class ServerJSONRemotePayload(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: str | None = None
    url: str | None = None


class MCPToolRequestPayload(BaseModel):
    name: str
    title: str | None = None
    description: str | None = None
    inputSchema: dict[str, Any] | None = None
    outputSchema: dict[str, Any] | None = None
    annotations: dict[str, Any] | None = None
    icons: list[MCPIconRequestPayload] | None = Field(
        default=None, max_length=_MAX_MCP_ICONS_PER_LIST
    )
    execution: dict[str, Any] | None = None


class MCPToolResponsePayload(BaseModel):
    name: str
    title: str | None = None
    description: str | None = None
    inputSchema: dict[str, Any] | None = None
    outputSchema: dict[str, Any] | None = None
    annotations: dict[str, Any] | None = None
    icons: list[MCPIconResponsePayload] | None = None
    execution: dict[str, Any] | None = None


class CreateMCPServerRequest(BaseModel):
    name: str
    description: str | None = None
    icons: list[MCPIconRequestPayload] | None = Field(
        default=None, max_length=_MAX_MCP_ICONS_PER_LIST
    )


class UpdateMCPServerRequest(BaseModel):
    display_name: str | None = None
    description: str | None = None
    icons: list[MCPIconRequestPayload] | None = Field(
        default=None, max_length=_MAX_MCP_ICONS_PER_LIST
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_latest_version_field(cls, value):
        if isinstance(value, dict) and "latest_version" in value:
            raise ValueError(
                "latest_version is read-only; it is resolved automatically "
                "from semantic-version ordering"
            )
        return value


class CreateMCPServerVersionRequest(BaseModel):
    server_json: ServerJSONPayload
    display_name: str | None = None
    status: str = "draft"
    source: str | None = None
    tools: list[MCPToolRequestPayload] | None = Field(
        default=None, max_length=_MAX_MCP_TOOLS_PER_LIST
    )


class UpdateMCPServerVersionRequest(BaseModel):
    display_name: str | None = None
    status: str | None = None
    tools: list[MCPToolRequestPayload] | None = Field(
        default=None, max_length=_MAX_MCP_TOOLS_PER_LIST
    )


class CreateMCPAccessEndpointRequest(BaseModel):
    server_version: str | None = None
    server_alias: str | None = None
    url: str
    transport_type: str = "streamable-http"


class UpdateMCPAccessEndpointRequest(BaseModel):
    server_version: str | None = None
    server_alias: str | None = None
    url: str | None = None
    transport_type: str | None = None


class SetAliasRequest(BaseModel):
    alias: str
    version: str


class SetTagRequest(BaseModel):
    key: str
    value: str


class AliasResponse(BaseModel):
    alias: str
    version: str


class MCPAccessEndpointSummaryResponse(BaseModel):
    id: str
    server_name: str
    url: str
    transport_type: str = "streamable-http"
    workspace: str | None = None
    server_version: str | None = None
    server_alias: str | None = None
    resolved_version: MCPServerVersionResponse | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    @classmethod
    def from_entity(cls, entity: MCPAccessEndpoint) -> MCPAccessEndpointSummaryResponse:
        return cls(
            id=entity.id,
            server_name=entity.server_name,
            url=entity.url,
            transport_type=str(entity.transport_type),
            workspace=entity.workspace,
            server_version=entity.server_version,
            server_alias=entity.server_alias,
            resolved_version=(
                None
                if entity.resolved_version is None
                else MCPServerVersionResponse.from_entity(entity.resolved_version)
            ),
            created_by=entity.created_by,
            last_updated_by=entity.last_updated_by,
            creation_timestamp=entity.creation_timestamp,
            last_updated_timestamp=entity.last_updated_timestamp,
        )


class MCPServerResponse(BaseModel):
    name: str
    display_name: str | None = None
    description: str | None = None
    icons: list[MCPIconResponsePayload] | None = None
    workspace: str | None = None
    status: str | None = None
    access_endpoints: list[MCPAccessEndpointSummaryResponse] = Field(default_factory=list)
    latest_version: str | None = None
    aliases: list[AliasResponse] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    @classmethod
    def from_entity(cls, entity: MCPServer) -> MCPServerResponse:
        return cls(
            name=entity.name,
            display_name=entity.display_name,
            description=entity.description,
            icons=entity.icons,
            workspace=entity.workspace,
            status=str(entity.status) if entity.status else None,
            access_endpoints=[
                MCPAccessEndpointSummaryResponse.from_entity(e) for e in entity.access_endpoints
            ],
            latest_version=entity.latest_version,
            aliases=[AliasResponse(alias=k, version=v) for k, v in entity.aliases.items()],
            tags=entity.tags,
            created_by=entity.created_by,
            last_updated_by=entity.last_updated_by,
            creation_timestamp=entity.creation_timestamp,
            last_updated_timestamp=entity.last_updated_timestamp,
        )


class MCPServerVersionResponse(BaseModel):
    name: str
    version: str
    server_json: dict[str, Any]
    display_name: str | None = None
    workspace: str | None = None
    status: str = "draft"
    tools: list[MCPToolResponsePayload] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    source: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    @classmethod
    def from_entity(cls, entity: MCPServerVersion) -> MCPServerVersionResponse:
        return cls(
            name=entity.name,
            version=entity.version,
            server_json=entity.server_json,
            display_name=entity.display_name,
            workspace=entity.workspace,
            status=str(entity.status),
            # Normalize unset tools to [] so clients can iterate without null guards.
            tools=[MCPToolResponsePayload(**t.to_dict()) for t in (entity.tools or [])],
            aliases=entity.aliases,
            tags=entity.tags,
            source=entity.source,
            created_by=entity.created_by,
            last_updated_by=entity.last_updated_by,
            creation_timestamp=entity.creation_timestamp,
            last_updated_timestamp=entity.last_updated_timestamp,
        )


class MCPAccessEndpointResponse(BaseModel):
    id: str
    server_name: str
    url: str
    transport_type: str = "streamable-http"
    workspace: str | None = None
    tools: list[MCPToolResponsePayload] | None = None
    server_version: str | None = None
    server_alias: str | None = None
    resolved_version: MCPServerVersionResponse | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    @classmethod
    def from_entity(cls, entity: MCPAccessEndpoint) -> MCPAccessEndpointResponse:
        tools = None
        if entity.resolved_version is not None and entity.resolved_version.tools is not None:
            tools = [MCPToolResponsePayload(**t.to_dict()) for t in entity.resolved_version.tools]
        return cls(
            id=entity.id,
            server_name=entity.server_name,
            url=entity.url,
            transport_type=str(entity.transport_type),
            workspace=entity.workspace,
            tools=tools,
            server_version=entity.server_version,
            server_alias=entity.server_alias,
            resolved_version=(
                None
                if entity.resolved_version is None
                else MCPServerVersionResponse.from_entity(entity.resolved_version)
            ),
            created_by=entity.created_by,
            last_updated_by=entity.last_updated_by,
            creation_timestamp=entity.creation_timestamp,
            last_updated_timestamp=entity.last_updated_timestamp,
        )


class SearchMCPServersResponse(BaseModel):
    mcp_servers: list[MCPServerResponse]
    next_page_token: str | None = None


class SearchMCPServerVersionsResponse(BaseModel):
    mcp_server_versions: list[MCPServerVersionResponse]
    next_page_token: str | None = None


class SearchMCPAccessEndpointsResponse(BaseModel):
    mcp_access_endpoints: list[MCPAccessEndpointResponse]
    next_page_token: str | None = None


def _parse_status(value: str | None) -> MCPStatus | None:
    if value is None:
        return None
    try:
        return MCPStatus(value)
    except ValueError:
        raise MlflowException.invalid_parameter_value(f"Invalid status: '{value}'") from None


def _parse_transport_type(value: str) -> MCPRemoteTransportType:
    try:
        return MCPRemoteTransportType(value)
    except ValueError:
        raise MlflowException.invalid_parameter_value(
            f"Invalid transport_type: '{value}'"
        ) from None


def _mlflow_error_response(e: MlflowException) -> JSONResponse:
    return JSONResponse(
        status_code=e.get_http_status_code(),
        content=json.loads(e.serialize_as_json()),
    )


def _format_validation_errors(exc: RequestValidationError) -> str:
    messages = []
    for error in exc.errors():
        if loc := ".".join(str(part) for part in error["loc"] if part != "body"):
            messages.append(f"{loc}: {error['msg']}")
        else:
            messages.append(str(error["msg"]))
    return "; ".join(messages)


def _request_validation_error_response(exc: RequestValidationError) -> JSONResponse:
    return _mlflow_error_response(
        MlflowException.invalid_parameter_value(
            f"Invalid request: {_format_validation_errors(exc)}"
        )
    )


def _tool_payloads_to_entities(tools: list[MCPToolRequestPayload] | None) -> list[MCPTool] | None:
    if tools is None:
        return None
    return [MCPTool.from_dict(t.model_dump(exclude_none=True)) for t in tools]


def _icon_payloads_to_entities(
    icons: list[MCPIconRequestPayload] | None,
) -> list[MCPIcon] | None:
    if icons is None:
        return None
    return [icon.model_dump(exclude_none=True) for icon in icons]


def _update_mcp_server_kwargs(name: str, body: UpdateMCPServerRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": name}
    provided_fields = body.model_fields_set
    for field_name in ("description", "display_name", "icons"):
        if field_name in provided_fields:
            kwargs[field_name] = (
                _icon_payloads_to_entities(body.icons)
                if field_name == "icons"
                else getattr(body, field_name)
            )
    return kwargs


def _update_mcp_server_version_kwargs(
    name: str, version: str, body: UpdateMCPServerVersionRequest
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": name, "version": version}
    provided_fields = body.model_fields_set
    if "display_name" in provided_fields:
        kwargs["display_name"] = body.display_name
    if "status" in provided_fields:
        if body.status is None:
            raise MlflowException.invalid_parameter_value(
                "status cannot be null; omit the field to leave it unchanged"
            )
        kwargs["status"] = _parse_status(body.status)
    if "tools" in provided_fields:
        kwargs["tools"] = _tool_payloads_to_entities(body.tools)
    return kwargs


def _ensure_version_create_parent_access(
    store, name: str, username: str | None, request: Request
) -> None:
    if not getattr(request.state, "mcp_server_parent_auto_created", False):
        return

    try:
        store.create_mcp_server(name=name, created_by=username)
    except MlflowException as e:
        if e.error_code != ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
            raise
        request.state.mcp_server_parent_auto_created = False
        can_update_existing = getattr(
            request.state, "mcp_server_can_update_existing_recheck", lambda: False
        )
        if not can_update_existing():
            raise MlflowException("Permission denied.", error_code=PERMISSION_DENIED)


def _update_mcp_access_endpoint_kwargs(
    server_name: str, endpoint_id: str, body: UpdateMCPAccessEndpointRequest
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"server_name": server_name, "endpoint_id": endpoint_id}
    provided_fields = body.model_fields_set
    for field_name in ("server_version", "server_alias", "url"):
        if field_name in provided_fields:
            kwargs[field_name] = getattr(body, field_name)
    if "transport_type" in provided_fields:
        kwargs["transport_type"] = (
            None if body.transport_type is None else _parse_transport_type(body.transport_type)
        )
    return kwargs


mcp_server_router = APIRouter(tags=["MCP Server Registry"])


@mcp_server_router.post("", response_model=MCPServerResponse)
def create_mcp_server(body: CreateMCPServerRequest, request: Request) -> MCPServerResponse:
    from mlflow.server.handlers import _get_tracking_store

    validate_mcp_server_name(body.name)
    username = getattr(request.state, "username", None)
    server = _get_tracking_store().create_mcp_server(
        name=body.name,
        description=body.description,
        icons=_icon_payloads_to_entities(body.icons),
        created_by=username,
    )
    return MCPServerResponse.from_entity(server)


@mcp_server_router.get("", response_model=SearchMCPServersResponse)
def search_mcp_servers(
    filter_string: str | None = Query(None),
    max_results: int = Query(100),
    order_by: list[str] | None = Query(None),
    page_token: str | None = Query(None),
) -> SearchMCPServersResponse:
    from mlflow.server.handlers import _get_tracking_store

    results = _get_tracking_store().search_mcp_servers(
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )
    return SearchMCPServersResponse(
        mcp_servers=[MCPServerResponse.from_entity(s) for s in results],
        next_page_token=results.token,
    )


# Static route — must be registered before /{name:path} routes
@mcp_server_router.get("/endpoints", response_model=SearchMCPAccessEndpointsResponse)
def search_all_access_endpoints(
    filter_string: str | None = Query(None),
    max_results: int = Query(100),
    order_by: list[str] | None = Query(None),
    page_token: str | None = Query(None),
    server_version: str | None = Query(None),
    server_alias: str | None = Query(None),
) -> SearchMCPAccessEndpointsResponse:
    from mlflow.server.handlers import _get_tracking_store

    store = _get_tracking_store()
    results = store.search_mcp_access_endpoints(
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
        server_version=server_version,
        server_alias=server_alias,
    )
    endpoints = [MCPAccessEndpointResponse.from_entity(e) for e in results]
    return SearchMCPAccessEndpointsResponse(
        mcp_access_endpoints=endpoints,
        next_page_token=results.token,
    )


@mcp_server_router.post("/{name:path}/versions/{version:path}/tags")
def set_mcp_server_version_tag(name: str, version: str, request: SetTagRequest) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _get_tracking_store().set_mcp_server_version_tag(
        name=name, version=version, key=request.key, value=request.value
    )
    return {}


@mcp_server_router.delete("/{name:path}/versions/{version:path}/tags/{key:path}")
def delete_mcp_server_version_tag(name: str, version: str, key: str) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _get_tracking_store().delete_mcp_server_version_tag(name=name, version=version, key=key)
    return {}


@mcp_server_router.get(
    "/{name:path}/versions/{version:path}",
    response_model=MCPServerVersionResponse,
)
def get_mcp_server_version(name: str, version: str) -> MCPServerVersionResponse:
    from mlflow.server.handlers import _get_tracking_store

    ver = _get_tracking_store().get_mcp_server_version(name, version)
    return MCPServerVersionResponse.from_entity(ver)


@mcp_server_router.patch(
    "/{name:path}/versions/{version:path}", response_model=MCPServerVersionResponse
)
def update_mcp_server_version(
    name: str, version: str, body: UpdateMCPServerVersionRequest, request: Request
) -> MCPServerVersionResponse:
    from mlflow.server.handlers import _get_tracking_store

    username = getattr(request.state, "username", None)
    ver = _get_tracking_store().update_mcp_server_version(
        **_update_mcp_server_version_kwargs(name, version, body), last_updated_by=username
    )
    return MCPServerVersionResponse.from_entity(ver)


@mcp_server_router.delete("/{name:path}/versions/{version:path}")
def delete_mcp_server_version(name: str, version: str) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _get_tracking_store().delete_mcp_server_version(name, version)
    return {}


@mcp_server_router.post("/{name:path}/versions", response_model=MCPServerVersionResponse)
def create_mcp_server_version(
    name: str, body: CreateMCPServerVersionRequest, request: Request
) -> MCPServerVersionResponse:
    from mlflow.server.handlers import _get_tracking_store

    validate_mcp_server_name(body.server_json.name)
    validate_mcp_server_name(name)
    if body.server_json.name != name:
        raise MlflowException.invalid_parameter_value(
            f"server_json.name '{body.server_json.name}' does not match path parameter '{name}'"
        )
    username = getattr(request.state, "username", None)
    status = _parse_status(body.status)
    tools = _tool_payloads_to_entities(body.tools)
    server_json = body.server_json.model_dump(by_alias=True, exclude_unset=True)
    store = _get_tracking_store()
    _ensure_version_create_parent_access(store, name, username, request)
    ver = store.create_mcp_server_version(
        server_json=server_json,
        display_name=body.display_name,
        source=body.source,
        status=status,
        tools=tools,
        created_by=username,
    )
    return MCPServerVersionResponse.from_entity(ver)


@mcp_server_router.get("/{name:path}/versions", response_model=SearchMCPServerVersionsResponse)
def search_mcp_server_versions(
    name: str,
    filter_string: str | None = Query(None),
    max_results: int = Query(100),
    order_by: list[str] | None = Query(None),
    page_token: str | None = Query(None),
) -> SearchMCPServerVersionsResponse:
    from mlflow.server.handlers import _get_tracking_store

    results = _get_tracking_store().search_mcp_server_versions(
        name=name,
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )
    return SearchMCPServerVersionsResponse(
        mcp_server_versions=[MCPServerVersionResponse.from_entity(v) for v in results],
        next_page_token=results.token,
    )


@mcp_server_router.post("/{name:path}/endpoints", response_model=MCPAccessEndpointResponse)
def create_mcp_access_endpoint(
    name: str, body: CreateMCPAccessEndpointRequest, request: Request
) -> MCPAccessEndpointResponse:
    from mlflow.server.handlers import _get_tracking_store

    username = getattr(request.state, "username", None)
    transport = _parse_transport_type(body.transport_type)
    store = _get_tracking_store()
    endpoint = store.create_mcp_access_endpoint(
        server_name=name,
        url=body.url,
        transport_type=transport,
        server_version=body.server_version,
        server_alias=body.server_alias,
        created_by=username,
    )
    return MCPAccessEndpointResponse.from_entity(endpoint)


@mcp_server_router.get(
    "/{name:path}/endpoints/{endpoint_id}",
    response_model=MCPAccessEndpointResponse,
)
def get_mcp_access_endpoint(name: str, endpoint_id: str) -> MCPAccessEndpointResponse:
    from mlflow.server.handlers import _get_tracking_store

    store = _get_tracking_store()
    endpoint = store.get_mcp_access_endpoint(name, endpoint_id)
    return MCPAccessEndpointResponse.from_entity(endpoint)


@mcp_server_router.patch(
    "/{name:path}/endpoints/{endpoint_id}",
    response_model=MCPAccessEndpointResponse,
)
def update_mcp_access_endpoint(
    name: str, endpoint_id: str, body: UpdateMCPAccessEndpointRequest, request: Request
) -> MCPAccessEndpointResponse:
    from mlflow.server.handlers import _get_tracking_store

    username = getattr(request.state, "username", None)
    store = _get_tracking_store()
    endpoint = store.update_mcp_access_endpoint(
        **_update_mcp_access_endpoint_kwargs(name, endpoint_id, body), last_updated_by=username
    )
    return MCPAccessEndpointResponse.from_entity(endpoint)


@mcp_server_router.delete("/{name:path}/endpoints/{endpoint_id}")
def delete_mcp_access_endpoint(name: str, endpoint_id: str) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _get_tracking_store().delete_mcp_access_endpoint(name, endpoint_id)
    return {}


@mcp_server_router.get("/{name:path}/endpoints", response_model=SearchMCPAccessEndpointsResponse)
def search_server_access_endpoints(
    name: str,
    filter_string: str | None = Query(None),
    max_results: int = Query(100),
    order_by: list[str] | None = Query(None),
    page_token: str | None = Query(None),
    server_version: str | None = Query(None),
    server_alias: str | None = Query(None),
) -> SearchMCPAccessEndpointsResponse:
    from mlflow.server.handlers import _get_tracking_store

    store = _get_tracking_store()
    results = store.search_mcp_access_endpoints(
        server_name=name,
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
        server_version=server_version,
        server_alias=server_alias,
    )
    endpoints = [MCPAccessEndpointResponse.from_entity(e) for e in results]
    return SearchMCPAccessEndpointsResponse(
        mcp_access_endpoints=endpoints,
        next_page_token=results.token,
    )


@mcp_server_router.post("/{name:path}/tags")
def set_mcp_server_tag(name: str, request: SetTagRequest) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _get_tracking_store().set_mcp_server_tag(name=name, key=request.key, value=request.value)
    return {}


@mcp_server_router.delete("/{name:path}/tags/{key:path}")
def delete_mcp_server_tag(name: str, key: str) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _get_tracking_store().delete_mcp_server_tag(name=name, key=key)
    return {}


@mcp_server_router.post("/{name:path}/aliases")
def set_mcp_server_alias(name: str, request: SetAliasRequest) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _get_tracking_store().set_mcp_server_alias(
        name=name, alias=request.alias, version=request.version
    )
    return {}


@mcp_server_router.get("/{name:path}/aliases/{alias:path}", response_model=MCPServerVersionResponse)
def get_version_by_alias(name: str, alias: str) -> MCPServerVersionResponse:
    from mlflow.server.handlers import _get_tracking_store

    version = _get_tracking_store().get_mcp_server_version_by_alias(name, alias)
    return MCPServerVersionResponse.from_entity(version)


@mcp_server_router.delete("/{name:path}/aliases/{alias:path}")
def delete_mcp_server_alias(name: str, alias: str) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _get_tracking_store().delete_mcp_server_alias(name=name, alias=alias)
    return {}


# Catch-all — must be registered last so {name:path} doesn't swallow sub-resource routes
@mcp_server_router.get("/{name:path}", response_model=MCPServerResponse)
def get_mcp_server(name: str) -> MCPServerResponse:
    from mlflow.server.handlers import _get_tracking_store

    server = _get_tracking_store().get_mcp_server(name)
    return MCPServerResponse.from_entity(server)


@mcp_server_router.patch("/{name:path}", response_model=MCPServerResponse)
def update_mcp_server(
    name: str, body: UpdateMCPServerRequest, request: Request
) -> MCPServerResponse:
    from mlflow.server.handlers import _get_tracking_store

    username = getattr(request.state, "username", None)
    server = _get_tracking_store().update_mcp_server(
        **_update_mcp_server_kwargs(name, body), last_updated_by=username
    )
    return MCPServerResponse.from_entity(server)


@mcp_server_router.delete("/{name:path}")
def delete_mcp_server(name: str) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _get_tracking_store().delete_mcp_server(name)
    return {}
