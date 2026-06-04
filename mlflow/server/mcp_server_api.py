from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, FastAPI, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from mlflow.entities.mcp_access_binding import MCPAccessBinding
from mlflow.entities.mcp_server import MCPRemoteTransportType, MCPServer, MCPStatus, MCPTool
from mlflow.entities.mcp_server_version import MCPServerVersion
from mlflow.exceptions import MlflowException

_MCP_SERVER_API_PREFIX = "/ajax-api/3.0/mlflow/mcp-servers"


class ServerJSONPayload(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str
    version: str
    title: str | None = None
    description: str | None = None
    packages: list[ServerJSONPackagePayload] | None = None
    remotes: list[ServerJSONRemotePayload] | None = None
    repository: str | None = None
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

    type: str
    url: str


class MCPToolPayload(BaseModel):
    name: str
    title: str | None = None
    description: str | None = None
    inputSchema: dict[str, Any] | None = None
    outputSchema: dict[str, Any] | None = None
    annotations: dict[str, Any] | None = None
    icons: list[dict[str, Any]] | None = None
    execution: dict[str, Any] | None = None


class CreateMCPServerRequest(BaseModel):
    name: str
    description: str | None = None
    icons: list[dict[str, Any]] | None = None


class UpdateMCPServerRequest(BaseModel):
    display_name: str | None = None
    description: str | None = None
    icons: list[dict[str, Any]] | None = None
    latest_version: str | None = None


class CreateMCPServerVersionRequest(BaseModel):
    server_json: ServerJSONPayload
    display_name: str | None = None
    status: str = "draft"
    source: str | None = None
    tools: list[MCPToolPayload] | None = None


class UpdateMCPServerVersionRequest(BaseModel):
    display_name: str | None = None
    status: str | None = None
    tools: list[MCPToolPayload] | None = None


class CreateMCPAccessBindingRequest(BaseModel):
    server_version: str | None = None
    server_alias: str | None = None
    endpoint_url: str
    transport_type: str = "streamable-http"


class UpdateMCPAccessBindingRequest(BaseModel):
    server_version: str | None = None
    server_alias: str | None = None
    endpoint_url: str | None = None
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


class MCPAccessBindingSummaryResponse(BaseModel):
    binding_id: int
    server_name: str
    endpoint_url: str
    transport_type: str = "streamable-http"
    server_version: str | None = None
    server_alias: str | None = None
    resolved_version: MCPServerVersionResponse | None = None

    @classmethod
    def from_entity(cls, entity: MCPAccessBinding) -> MCPAccessBindingSummaryResponse:
        return cls(
            binding_id=entity.binding_id,
            server_name=entity.server_name,
            endpoint_url=entity.endpoint_url,
            transport_type=str(entity.transport_type),
            server_version=entity.server_version,
            server_alias=entity.server_alias,
            resolved_version=(
                None
                if entity.resolved_version is None
                else MCPServerVersionResponse.from_entity(entity.resolved_version)
            ),
        )


class MCPServerResponse(BaseModel):
    name: str
    display_name: str | None = None
    description: str | None = None
    icons: list[dict[str, Any]] | None = None
    status: str | None = None
    access_bindings: list[MCPAccessBindingSummaryResponse] = Field(default_factory=list)
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
            status=str(entity.status) if entity.status else None,
            access_bindings=[
                MCPAccessBindingSummaryResponse.from_entity(b) for b in entity.access_bindings
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
    status: str = "draft"
    tools: list[MCPToolPayload] | None = None
    aliases: list[str] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    source: str | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    @classmethod
    def from_entity(cls, entity: MCPServerVersion) -> MCPServerVersionResponse:
        tools = None
        if entity.tools is not None:
            tools = [MCPToolPayload(**t.to_dict()) for t in entity.tools]
        return cls(
            name=entity.name,
            version=entity.version,
            server_json=entity.server_json,
            display_name=entity.display_name,
            status=str(entity.status),
            tools=tools,
            aliases=entity.aliases,
            tags=entity.tags,
            source=entity.source,
            created_by=entity.created_by,
            last_updated_by=entity.last_updated_by,
            creation_timestamp=entity.creation_timestamp,
            last_updated_timestamp=entity.last_updated_timestamp,
        )


class MCPAccessBindingResponse(BaseModel):
    binding_id: int
    server_name: str
    endpoint_url: str
    transport_type: str = "streamable-http"
    tools: list[MCPToolPayload] | None = None
    server_version: str | None = None
    server_alias: str | None = None
    resolved_version: MCPServerVersionResponse | None = None
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    @classmethod
    def from_entity(cls, entity: MCPAccessBinding) -> MCPAccessBindingResponse:
        tools = None
        if entity.resolved_version is not None and entity.resolved_version.tools is not None:
            tools = [MCPToolPayload(**t.to_dict()) for t in entity.resolved_version.tools]
        return cls(
            binding_id=entity.binding_id,
            server_name=entity.server_name,
            endpoint_url=entity.endpoint_url,
            transport_type=str(entity.transport_type),
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


class SearchMCPAccessBindingsResponse(BaseModel):
    mcp_access_bindings: list[MCPAccessBindingResponse]
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


def add_mcp_server_exception_handlers(fastapi_app: FastAPI) -> None:
    if getattr(fastapi_app.state, "mcp_server_exception_handlers_added", False):
        return

    @fastapi_app.exception_handler(MlflowException)
    async def mlflow_exception_handler(_request, exc: MlflowException):
        return _mlflow_error_response(exc)

    @fastapi_app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(_request, exc: RequestValidationError):
        return _request_validation_error_response(exc)

    fastapi_app.state.mcp_server_exception_handlers_added = True


def _tool_payloads_to_entities(tools: list[MCPToolPayload] | None) -> list[MCPTool] | None:
    if tools is None:
        return None
    return [MCPTool.from_dict(t.model_dump(exclude_none=True)) for t in tools]


def _update_mcp_server_kwargs(name: str, request: UpdateMCPServerRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": name}
    provided_fields = request.model_fields_set
    for field_name in ("description", "display_name", "icons", "latest_version"):
        if field_name in provided_fields:
            kwargs[field_name] = getattr(request, field_name)
    return kwargs


def _update_mcp_server_version_kwargs(
    name: str, version: str, request: UpdateMCPServerVersionRequest
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": name, "version": version}
    provided_fields = request.model_fields_set
    if "display_name" in provided_fields:
        kwargs["display_name"] = request.display_name
    if "status" in provided_fields:
        kwargs["status"] = None if request.status is None else _parse_status(request.status)
    if "tools" in provided_fields:
        kwargs["tools"] = _tool_payloads_to_entities(request.tools)
    return kwargs


def _update_mcp_access_binding_kwargs(
    server_name: str, binding_id: int, request: UpdateMCPAccessBindingRequest
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"server_name": server_name, "binding_id": binding_id}
    provided_fields = request.model_fields_set
    for field_name in ("server_version", "server_alias", "endpoint_url"):
        if field_name in provided_fields:
            kwargs[field_name] = getattr(request, field_name)
    if "transport_type" in provided_fields:
        kwargs["transport_type"] = (
            None
            if request.transport_type is None
            else _parse_transport_type(request.transport_type)
        )
    return kwargs


mcp_server_router = APIRouter(tags=["MCP Server Registry"])


@mcp_server_router.post("", response_model=MCPServerResponse)
def create_mcp_server(request: CreateMCPServerRequest) -> MCPServerResponse:
    from mlflow.server.handlers import _get_tracking_store

    server = _get_tracking_store().create_mcp_server(
        name=request.name,
        description=request.description,
        icons=request.icons,
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
@mcp_server_router.get("/bindings", response_model=SearchMCPAccessBindingsResponse)
def search_all_access_bindings(
    filter_string: str | None = Query(None),
    max_results: int = Query(100),
    order_by: list[str] | None = Query(None),
    page_token: str | None = Query(None),
    server_version: str | None = Query(None),
    server_alias: str | None = Query(None),
) -> SearchMCPAccessBindingsResponse:
    from mlflow.server.handlers import _get_tracking_store

    store = _get_tracking_store()
    results = store.search_mcp_access_bindings(
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
        server_version=server_version,
        server_alias=server_alias,
    )
    bindings = [MCPAccessBindingResponse.from_entity(b) for b in results]
    return SearchMCPAccessBindingsResponse(
        mcp_access_bindings=bindings,
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
    name: str, version: str, request: UpdateMCPServerVersionRequest
) -> MCPServerVersionResponse:
    from mlflow.server.handlers import _get_tracking_store

    ver = _get_tracking_store().update_mcp_server_version(
        **_update_mcp_server_version_kwargs(name, version, request)
    )
    return MCPServerVersionResponse.from_entity(ver)


@mcp_server_router.delete("/{name:path}/versions/{version:path}")
def delete_mcp_server_version(name: str, version: str) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _get_tracking_store().delete_mcp_server_version(name, version)
    return {}


@mcp_server_router.post("/{name:path}/versions", response_model=MCPServerVersionResponse)
def create_mcp_server_version(
    name: str, request: CreateMCPServerVersionRequest
) -> MCPServerVersionResponse:
    from mlflow.server.handlers import _get_tracking_store

    if request.server_json.name != name:
        raise MlflowException.invalid_parameter_value(
            f"server_json.name '{request.server_json.name}' does not match path parameter '{name}'"
        )
    status = _parse_status(request.status)
    tools = _tool_payloads_to_entities(request.tools)
    server_json = request.server_json.model_dump(by_alias=True, exclude_unset=True)
    ver = _get_tracking_store().create_mcp_server_version(
        server_json=server_json,
        display_name=request.display_name,
        source=request.source,
        status=status,
        tools=tools,
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


@mcp_server_router.post("/{name:path}/bindings", response_model=MCPAccessBindingResponse)
def create_mcp_access_binding(
    name: str, request: CreateMCPAccessBindingRequest
) -> MCPAccessBindingResponse:
    from mlflow.server.handlers import _get_tracking_store

    transport = _parse_transport_type(request.transport_type)
    store = _get_tracking_store()
    binding = store.create_mcp_access_binding(
        server_name=name,
        endpoint_url=request.endpoint_url,
        transport_type=transport,
        server_version=request.server_version,
        server_alias=request.server_alias,
    )
    return MCPAccessBindingResponse.from_entity(binding)


@mcp_server_router.get(
    "/{name:path}/bindings/{binding_id}",
    response_model=MCPAccessBindingResponse,
)
def get_mcp_access_binding(name: str, binding_id: int) -> MCPAccessBindingResponse:
    from mlflow.server.handlers import _get_tracking_store

    store = _get_tracking_store()
    binding = store.get_mcp_access_binding(name, binding_id)
    return MCPAccessBindingResponse.from_entity(binding)


@mcp_server_router.patch(
    "/{name:path}/bindings/{binding_id}",
    response_model=MCPAccessBindingResponse,
)
def update_mcp_access_binding(
    name: str, binding_id: int, request: UpdateMCPAccessBindingRequest
) -> MCPAccessBindingResponse:
    from mlflow.server.handlers import _get_tracking_store

    store = _get_tracking_store()
    binding = store.update_mcp_access_binding(
        **_update_mcp_access_binding_kwargs(name, binding_id, request)
    )
    return MCPAccessBindingResponse.from_entity(binding)


@mcp_server_router.delete("/{name:path}/bindings/{binding_id}")
def delete_mcp_access_binding(name: str, binding_id: int) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _get_tracking_store().delete_mcp_access_binding(name, binding_id)
    return {}


@mcp_server_router.get("/{name:path}/bindings", response_model=SearchMCPAccessBindingsResponse)
def search_server_access_bindings(
    name: str,
    filter_string: str | None = Query(None),
    max_results: int = Query(100),
    order_by: list[str] | None = Query(None),
    page_token: str | None = Query(None),
    server_version: str | None = Query(None),
    server_alias: str | None = Query(None),
) -> SearchMCPAccessBindingsResponse:
    from mlflow.server.handlers import _get_tracking_store

    store = _get_tracking_store()
    results = store.search_mcp_access_bindings(
        server_name=name,
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
        server_version=server_version,
        server_alias=server_alias,
    )
    bindings = [MCPAccessBindingResponse.from_entity(b) for b in results]
    return SearchMCPAccessBindingsResponse(
        mcp_access_bindings=bindings,
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
def update_mcp_server(name: str, request: UpdateMCPServerRequest) -> MCPServerResponse:
    from mlflow.server.handlers import _get_tracking_store

    server = _get_tracking_store().update_mcp_server(**_update_mcp_server_kwargs(name, request))
    return MCPServerResponse.from_entity(server)


@mcp_server_router.delete("/{name:path}")
def delete_mcp_server(name: str) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _get_tracking_store().delete_mcp_server(name)
    return {}
