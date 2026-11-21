from __future__ import annotations

import contextvars
import threading
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from opentelemetry import trace as otel_trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span as SDKSpan
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.trace import SpanKind

from mlflow.entities.mcp_server_ref import MCPServerRef
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    from fastmcp import FastMCP
    from fastmcp.server.middleware import CallNext, MiddlewareContext

__all__ = ["instrument_fastmcp"]


@dataclass
class _CallState:
    identity: MCPServerRef
    active: bool = True
    annotated: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


_active_call: contextvars.ContextVar[_CallState | None] = contextvars.ContextVar(
    "mlflow_fastmcp_active_call", default=None
)
_registration_lock = threading.Lock()
_server_registrations: weakref.WeakKeyDictionary[Any, tuple[TracerProvider, MCPServerRef]] = (
    weakref.WeakKeyDictionary()
)
_provider_processors: weakref.WeakKeyDictionary[TracerProvider, _MCPServerIdentitySpanProcessor] = (
    weakref.WeakKeyDictionary()
)


class _MCPServerIdentitySpanProcessor(SpanProcessor):
    def __init__(self, instrumentation_scope: str) -> None:
        self._instrumentation_scope = instrumentation_scope

    def on_start(self, span: SDKSpan, parent_context: Context | None = None) -> None:
        call_state = _active_call.get()
        if call_state is None or span.kind != SpanKind.SERVER:
            return

        scope = getattr(span, "instrumentation_scope", None)
        if scope is None:
            scope = getattr(span, "instrumentation_info", None)
        if getattr(scope, "name", None) != self._instrumentation_scope:
            return

        # FastMCP sets gen_ai.tool.name only after the span starts, so read the tool name
        # from the span name. It names the tool that actually ran, even if a middleware
        # renamed the requested tool.
        name_prefix = "tools/call "
        if not span.name.startswith(name_prefix):
            return
        tool_name = span.name[len(name_prefix) :]
        if not tool_name:
            return

        with call_state.lock:
            if not call_state.active or call_state.annotated:
                return

            span.set_attribute(SpanAttributeKey.MCP_SERVER_NAME, call_state.identity.name)
            span.set_attribute(SpanAttributeKey.MCP_TOOL_NAME, tool_name)
            if call_state.identity.version is not None:
                span.set_attribute(SpanAttributeKey.MCP_SERVER_VERSION, call_state.identity.version)
            call_state.annotated = True


@experimental(version="3.19.0")
def instrument_fastmcp(
    server: FastMCP,
    *,
    name: str,
    version: str | None = None,
) -> None:
    """Add MCP Registry identity to a FastMCP server's native tool-call spans.

    Call this during server setup, before starting the server and after configuring the
    global OpenTelemetry SDK provider with ``opentelemetry.trace.set_tracer_provider()``.
    FastMCP creates its spans through that provider. Multiple servers may share it.

    Args:
        server: FastMCP server instance to instrument.
        name: Registered MCP server name, such as ``com.example/weather``.
        version: Optional concrete version running on this deployment.

    Raises:
        MlflowException: If the server identity or provider configuration is invalid,
            or if the server has already been registered with a different identity or provider.
    """
    identity = MCPServerRef(name=name, version=version)

    if not callable(getattr(server, "add_middleware", None)):
        raise MlflowException.invalid_parameter_value(
            "FastMCP server tracing requires a FastMCP server with add_middleware() support "
            "(FastMCP>=3.4.8). The `mlflow[mcp]` extra allows older versions for client use; "
            "install a compatible version with `pip install 'mlflow[mcp]' "
            "'fastmcp>=3.4.8,<4'`."
        )

    try:
        from fastmcp.server.middleware import Middleware
        from fastmcp.telemetry import INSTRUMENTATION_NAME
    except ImportError as exc:
        raise MlflowException.invalid_parameter_value(
            "FastMCP server tracing requires FastMCP>=3.4.8 with server middleware and native "
            "OpenTelemetry telemetry. The `mlflow[mcp]` extra allows older versions for client "
            "use; install a compatible version with `pip install 'mlflow[mcp]' "
            "'fastmcp>=3.4.8,<4'`."
        ) from exc

    provider = otel_trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        raise MlflowException.invalid_parameter_value(
            "FastMCP server tracing requires an initialized OpenTelemetry SDK TracerProvider. "
            "Configure one with opentelemetry.trace.set_tracer_provider() before calling "
            "instrument_fastmcp()."
        )

    with _registration_lock:
        try:
            registration = _server_registrations.get(server)
        except TypeError:
            raise MlflowException.invalid_parameter_value(
                "server must be a hashable, weak-referenceable FastMCP server instance."
            ) from None

        if registration is not None:
            registered_provider, registered_identity = registration
            if registered_provider is provider and registered_identity == identity:
                return
            raise MlflowException.invalid_parameter_value(
                "This FastMCP server is already instrumented with a different provider or "
                "MCP Registry identity."
            )

        if provider not in _provider_processors:
            processor = _MCPServerIdentitySpanProcessor(INSTRUMENTATION_NAME)
            provider.add_span_processor(processor)
            _provider_processors[provider] = processor

        class _MCPServerIdentityMiddleware(Middleware):
            async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext) -> Any:
                call_state = _CallState(identity=identity)
                token = _active_call.set(call_state)
                try:
                    return await call_next(context)
                finally:
                    with call_state.lock:
                        call_state.active = False
                    _active_call.reset(token)

        server.add_middleware(_MCPServerIdentityMiddleware())
        _server_registrations[server] = (provider, identity)
