from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Generator


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


@contextlib.contextmanager
def context(
    metadata: dict[str, str] | None = None,
    tags: dict[str, str] | None = None,
    enabled: bool | None = None,
) -> Generator[None, None, None]:
    """
    A context manager that injects metadata and/or tags into any trace created
    within its scope, without creating a wrapper span. It can also be used to
    selectively disable tracing within the scope.

    This is useful when you need to attach trace-level information (e.g. session
    IDs) to traces produced by code you don't control like auto-instrumented libraries,
    or when you want to suppress tracing for a specific code block without affecting
    the global tracing state.

    .. code-block:: python

        import mlflow

        # Enable auto-tracing for LangChain
        mlflow.langchain.autolog()

        with mlflow.tracing.context(
            # Specify metadata and tags you want to inject into the trace
            tags={"project": "my-project"},
            # Reserved metadata key for associating traces with a conversation session.
            metadata={"mlflow.trace.session": "session-123"},
        ):
            # Any trace created inside this block will carry the metadata and tags.
            agent.invoke("What is the capital of France?")

        # Disable tracing within a specific block
        with mlflow.tracing.context(enabled=False):
            # No traces will be created inside this block.
            agent.invoke("This call will not be traced")

    The context manager can be nested to combine multiple sets of metadata and tags. When same key
    is specified in multiple levels, the value from the inner level takes precedence.

    .. code-block:: python

        import mlflow

        with mlflow.tracing.context(metadata={"foo": "bar", "baz": "qux"}):
            with mlflow.tracing.context(metadata={"foo": "baz", "qux": "quux"}):
                my_func()

        # Trace created by my_func will have metadata={"foo": "baz", "baz": "qux", "qux": "quux"}

    Args:
        metadata: Key-value pairs to inject into the trace's ``request_metadata``
            (immutable after trace creation).
        tags: Key-value pairs to inject into the trace's ``tags``.
        enabled: Whether tracing is enabled within the scope. If ``False``, all
            tracing calls within the scope will return ``NoOpSpan`` without creating
            any traces. If ``None`` (default), the value is inherited from the outer
            scope. This does not affect the global tracing state set by
            :py:func:`mlflow.tracing.disable`.
    """
    current = _USER_TRACE_CONTEXT.get()

    # Merge with any outer context scope
    merged_metadata = {**(current.metadata if current else {}), **(metadata or {})}
    merged_tags = {**(current.tags if current else {}), **(tags or {})}
    resolved_enabled = enabled if enabled is not None else (current.enabled if current else None)

    token = _USER_TRACE_CONTEXT.set(
        _UserTraceContext(metadata=merged_metadata, tags=merged_tags, enabled=resolved_enabled)
    )
    try:
        yield
    finally:
        _USER_TRACE_CONTEXT.reset(token)
