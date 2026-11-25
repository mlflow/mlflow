import asyncio
import hashlib
import logging
import os
from concurrent.futures import Future
from contextlib import asynccontextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from functools import wraps
from threading import BoundedSemaphore, Thread
from typing import TYPE_CHECKING
from uuid import uuid4
from weakref import ref

from packaging.version import Version

import mlflow
from mlflow.entities import LiveSpan, MCPServerRef
from mlflow.entities.mcp_server import MCPRemoteTransportType
from mlflow.langchain.constant import FLAVOR_NAME
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils import get_installed_version, gorilla
from mlflow.utils.autologging_utils.safety import _store_patch, _wrap_patch, safe_patch
from mlflow.utils.rest_utils import _single_attempt_http_requests
from mlflow.utils.workspace_context import ServerWorkspaceContext, get_request_workspace

if TYPE_CHECKING:
    from fastmcp import Client
    from fastmcp.client.transports import SSETransport, StreamableHttpTransport
    from mcp import ClientSession

_logger = logging.getLogger(__name__)
_MCP_TOOL_METADATA_KEY = "_mlflow_mcp_tool"
_LOOKUP_TIMEOUT_SECONDS = 2.0
_MAX_CONCURRENT_LOOKUPS = 32
_lookup_slots = BoundedSemaphore(_MAX_CONCURRENT_LOOKUPS)
# Concurrent tool calls share a pending workspace lookup. A finished lookup is not reused,
# since an experiment can move to another workspace.
_pending_workspace_lookups: dict[tuple[object, ...], asyncio.Task] = {}


def _reset_lookup_slots() -> None:
    global _lookup_slots
    _lookup_slots = BoundedSemaphore(_MAX_CONCURRENT_LOOKUPS)
    _pending_workspace_lookups.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_lookup_slots)


@dataclass(frozen=True)
class _MCPToolOrigin:
    server: MCPServerRef
    tracking_uri: str
    workspace: str


def _url_digest(url: str) -> bytes:
    return hashlib.sha256(url.encode()).digest()


class _MCPToolMetadata(str):
    # Other LangChain callbacks can serialize the opaque ID without seeing the connection.
    __slots__ = ("origin", "tool_name", "_client", "_transport", "_url_digest")

    def __new__(
        cls,
        origin: _MCPToolOrigin,
        client: "Client",
        transport: "SSETransport | StreamableHttpTransport",
        url: str,
        tool_name: str,
    ):
        marker = super().__new__(cls, uuid4().hex)
        marker.origin = origin
        marker.tool_name = tool_name
        marker._client = ref(client)
        marker._transport = ref(transport)
        marker._url_digest = _url_digest(url)
        return marker

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return str, (str(self),)

    def session_matches(self, session: "ClientSession", url_digest: bytes) -> bool:
        client = self._client()
        transport = self._transport()
        return (
            client is not None
            and transport is not None
            and client.transport is transport
            and client.is_connected()
            and client.session is session
            and url_digest == self._url_digest
        )


@dataclass(frozen=True)
class _PendingToolLink:
    span: LiveSpan
    marker: _MCPToolMetadata
    tracking_uri: str
    experiment_id: str | None
    workspace: str | None


@dataclass
class _MCPDiscovery:
    client: "Client"
    url_digest: bytes | None = None


_MCP_DISCOVERY: ContextVar[_MCPDiscovery | None] = ContextVar("_MCP_DISCOVERY", default=None)


def patch_mcp_adapter() -> None:
    # The adapter API is beta. Update this range when testing newer releases.
    langchain_version = get_installed_version("langchain")
    if (
        langchain_version is None
        or langchain_version.is_prerelease
        or not Version("1.4.0") <= langchain_version <= Version("1.5.0")
    ):
        return

    try:
        from fastmcp.client.transports import SSETransport, StreamableHttpTransport
        from langchain_core._api import suppress_langchain_beta_warning

        with suppress_langchain_beta_warning():
            from langchain.mcp import MCPAdapter
    except ImportError:
        return

    for transport_class in (StreamableHttpTransport, SSETransport):
        _patch_transport_sessions(transport_class)
    safe_patch(FLAVOR_NAME, MCPAdapter, "__init__", _patched_mcp_adapter_init, observe_only=True)
    safe_patch(FLAVOR_NAME, MCPAdapter, "__aenter__", _patched_mcp_adapter_enter, observe_only=True)
    safe_patch(FLAVOR_NAME, MCPAdapter, "list_tools", _patched_list_tools, observe_only=True)


def _patched_mcp_adapter_init(original, self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    transport = getattr(self.client, "transport", None)
    if _remote_transport_type(transport) is not None:
        self._mlflow_mcp_url = transport.url
        transport._mlflow_track_mcp_session = True
    return result


def _remote_transport_type(transport) -> MCPRemoteTransportType | None:
    from fastmcp.client.transports import SSETransport, StreamableHttpTransport

    if isinstance(transport, StreamableHttpTransport):
        return MCPRemoteTransportType.STREAMABLE_HTTP
    if isinstance(transport, SSETransport):
        return MCPRemoteTransportType.SSE
    return None


async def _patched_mcp_adapter_enter(original, self, *args, **kwargs):
    result = await original(self, *args, **kwargs)
    if (discovery := _MCP_DISCOVERY.get()) is not None and discovery.client is self.client:
        if self.client.is_connected():
            discovery.url_digest = getattr(self.client.session, "_mlflow_mcp_url_digest", None)
    return result


def _patch_transport_sessions(transport_class) -> None:
    original = gorilla.get_original_attribute(transport_class, "connect_session")

    @wraps(original)
    def connect_session(transport, *args, **kwargs):
        connection = original(transport, *args, **kwargs)
        if getattr(transport, "_mlflow_track_mcp_session", False):
            return _tracked_transport_session(transport, connection)
        return connection

    # FastMCP clients can share or copy a transport. Keep method binding on the class.
    _store_patch(FLAVOR_NAME, _wrap_patch(transport_class, "connect_session", connect_session))


@asynccontextmanager
async def _tracked_transport_session(transport, connection):
    # The configured URL can change while this session remains connected.
    digest = _url_digest(transport.url)
    async with connection as session:
        session._mlflow_mcp_url_digest = digest
        call_tool = session.call_tool

        @wraps(call_tool)
        async def tracked_call(name, *args, **kwargs):
            async with _link_session_tool_call(session, digest, name):
                return await call_tool(name, *args, **kwargs)

        session.call_tool = tracked_call
        yield session


def _find_mcp_tool_origin(
    tracking_uri: str, url: str, transport: MCPRemoteTransportType, workspace: str | None
) -> _MCPToolOrigin | None:
    match = None
    page_token = None
    with ServerWorkspaceContext(workspace):
        store = _get_store(store_uri=tracking_uri)
        while True:
            page = store.search_mcp_access_endpoints(page_token=page_token)
            for endpoint in page:
                if endpoint.url == url and endpoint.transport_type == transport:
                    if match is not None:
                        return None
                    match = endpoint
            if not (page_token := page.token):
                break

    if match is None:
        return None

    resolved_version = match.resolved_version
    return _MCPToolOrigin(
        MCPServerRef(
            match.server_name, version=resolved_version.version if resolved_version else None
        ),
        tracking_uri,
        match.workspace,
    )


async def _patched_list_tools(original, self, *args, **kwargs):
    if not (url := getattr(self, "_mlflow_mcp_url", None)):
        return await original(self, *args, **kwargs)

    # Use the connection's transport, since a URL can be used for HTTP or SSE.
    client = self.client
    connection = getattr(client, "transport", None)
    transport = _remote_transport_type(connection)
    if transport is None:
        return await original(self, *args, **kwargs)

    tracking_uri = mlflow.get_tracking_uri()
    workspace = get_request_workspace()
    discovery = _MCPDiscovery(client)
    token = _MCP_DISCOVERY.set(discovery)
    try:
        tools = await original(self, *args, **kwargs)
    finally:
        _MCP_DISCOVERY.reset(token)

    if not tools or discovery.url_digest != _url_digest(url):
        return tools

    origin = await _lookup(_find_mcp_tool_origin, tracking_uri, url, transport, workspace)
    if origin is not None:
        for tool in tools:
            # LangChain can expose the tool under a different name after discovery.
            marker = _MCPToolMetadata(origin, client, connection, url, tool_name=tool.name)
            tool.metadata = {**(tool.metadata or {}), _MCP_TOOL_METADATA_KEY: marker}
    return tools


def _is_in_memory_sqlite(tracking_uri: str) -> bool:
    if not tracking_uri.startswith(("sqlite:", "sqlite+")):
        return False

    from sqlalchemy.engine import make_url
    from sqlalchemy.util import asbool

    url = make_url(tracking_uri)
    return url.database in (None, "", ":memory:") or (
        asbool(url.query.get("uri", False))
        and (url.database == "file::memory:" or url.query.get("mode") == "memory")
    )


async def _lookup(function, tracking_uri: str, *args):
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _LOOKUP_TIMEOUT_SECONDS
        with _single_attempt_http_requests(_LOOKUP_TIMEOUT_SECONDS):
            if _is_in_memory_sqlite(tracking_uri):
                # The default SQLite pool gives each thread a separate in-memory database.
                return function(tracking_uri, *args)

            # Keep blocking I/O out of the event loop's executor. A timed-out lookup can
            # still be unwinding in the client, so limit the number of background workers.
            slots = _lookup_slots
            while (remaining := deadline - loop.time()) > 0:
                if slots.acquire(blocking=False):
                    break
                # The slots are shared by all threads and event loops in this process.
                await asyncio.sleep(min(remaining, 0.01))
            else:
                return None

            result = Future()

            def run():
                try:
                    result.set_result(function(tracking_uri, *args))
                except BaseException as exc:
                    result.set_exception(exc)
                finally:
                    slots.release()

            result.set_running_or_notify_cancel()
            try:
                Thread(
                    target=copy_context().run, args=(run,), name="mlflow-mcp-lookup", daemon=True
                ).start()
            except Exception:
                slots.release()
                raise

            return await asyncio.wait_for(
                asyncio.wrap_future(result), timeout=max(0, deadline - loop.time())
            )
    except Exception:
        _logger.debug("Failed to look up LangChain MCP trace-linking metadata.")
        return None


def _experiment_workspace(tracking_uri: str, experiment_id: str) -> str:
    with ServerWorkspaceContext(None):
        return _get_store(store_uri=tracking_uri).get_experiment(experiment_id).workspace


def _default_workspace(
    tracking_uri: str, url: str, transport: MCPRemoteTransportType
) -> str | None:
    # Without a workspace, the tracking server applies its default, as it does when linking a
    # span manually. Find the workspace where that default binds this endpoint.
    origin = _find_mcp_tool_origin(tracking_uri, url, transport, None)
    return origin.workspace if origin is not None else None


def _pending_workspace(link: _PendingToolLink) -> asyncio.Task:
    if link.experiment_id is not None:
        function = _experiment_workspace
        args = (link.experiment_id,)
    else:
        # Traces exported only through OTLP have no experiment.
        transport = link.marker._transport()
        function = _default_workspace
        args = (transport.url, _remote_transport_type(transport))
    loop = asyncio.get_running_loop()
    key = (loop, function, link.tracking_uri, *args)
    if (task := _pending_workspace_lookups.get(key)) is None:
        task = loop.create_task(_lookup(function, link.tracking_uri, *args))
        _pending_workspace_lookups[key] = task
        task.add_done_callback(lambda _: _pending_workspace_lookups.pop(key, None))
    return task


@asynccontextmanager
async def _link_session_tool_call(session: "ClientSession", url_digest: bytes, name: str):
    links = []
    try:
        for link in _take_session_tool_links(session, url_digest, name):
            # Resolve an implicit workspace alongside the MCP request, so the lookup does not
            # delay sending the request.
            lookup = _pending_workspace(link) if link.workspace is None else None
            links.append((link, lookup))
    except Exception:
        _logger.debug("Failed to prepare LangChain MCP tool span links.")

    cancelled = False
    try:
        yield
    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        for link, lookup in links:
            # Other tool calls can be waiting on the same lookup, so leave it running. Its
            # deadline bounds how long it holds a worker slot.
            if cancelled:
                continue
            try:
                # A lookup still pending when the tool returns is bounded by its deadline.
                workspace = await asyncio.shield(lookup) if lookup is not None else link.workspace
                if workspace == link.marker.origin.workspace:
                    link.span.link_mcp_server(link.marker.origin.server, tool=name)
            except Exception:
                _logger.debug("Failed to link the LangChain MCP tool span.")


def _take_session_tool_links(session: "ClientSession", url_digest: bytes, name: str):
    from langchain_core.runnables.config import ensure_config

    from mlflow.langchain.langchain_tracer import MlflowLangchainTracer

    # The child callback manager carries the tool run ID even when the tracer ran in a thread.
    callbacks = ensure_config().get("callbacks")
    if (run_id := getattr(callbacks, "parent_run_id", None)) is None:
        return
    for handler in callbacks.handlers:
        if not isinstance(handler, MlflowLangchainTracer):
            continue
        link = handler._mcp_tool_links.get(run_id)
        if (
            link is not None
            and link.marker.tool_name == name
            and link.marker.session_matches(session, url_digest)
        ):
            handler._mcp_tool_links.pop(run_id, None)
            yield link


def pending_mcp_tool_link(span: LiveSpan, marker: _MCPToolMetadata) -> _PendingToolLink | None:
    with InMemoryTraceManager.get_instance().get_trace(span.trace_id) as trace:
        if trace is None or not trace.tracking_uri:
            return None
        tracking_uri = trace.tracking_uri
        workspace = trace.workspace
        experiment_id = trace.info.experiment_id

    if tracking_uri != marker.origin.tracking_uri:
        return None
    return _PendingToolLink(span, marker, tracking_uri, experiment_id, workspace)
