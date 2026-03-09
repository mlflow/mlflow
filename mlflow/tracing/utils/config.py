from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass(frozen=True)
class _UserTraceContext:
    """
    Metadata and tags declared via ``mlflow.tracing.context()`` that should be
    injected into every trace created within the current scope.
    """

    metadata: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    enabled: bool | None = None


_USER_TRACE_CONTEXT: ContextVar[_UserTraceContext | None] = ContextVar(
    "mlflow_user_trace_context", default=None
)


def get_configured_trace_metadata() -> dict[str, str] | None:
    info = _USER_TRACE_CONTEXT.get()
    return info.metadata or None if info else None


def get_configured_trace_tags() -> dict[str, str] | None:
    info = _USER_TRACE_CONTEXT.get()
    return info.tags or None if info else None
