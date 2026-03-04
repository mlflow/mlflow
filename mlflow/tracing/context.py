from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass(frozen=True)
class _ConfiguredTraceInfo:
    """Metadata and tags declared via ``mlflow.configure_trace()`` that should be
    injected into every trace created within the current scope."""

    metadata: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)


_CONFIGURE_TRACE_INFO: ContextVar[_ConfiguredTraceInfo | None] = ContextVar(
    "mlflow_configure_trace_info", default=None
)


def get_configured_trace_metadata() -> dict[str, str] | None:
    info = _CONFIGURE_TRACE_INFO.get()
    return info.metadata or None if info else None


def get_configured_trace_tags() -> dict[str, str] | None:
    info = _CONFIGURE_TRACE_INFO.get()
    return info.tags or None if info else None
