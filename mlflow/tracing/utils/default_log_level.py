from __future__ import annotations

from mlflow.entities.span_log_level import SpanLogLevel

# Span types whose default level is INFO. These represent the user-visible
# semantic operations (model calls, tool calls, retrievals, agent turns, etc.)
# that should remain visible at the default UI threshold. Everything else —
# chain glue, output parsing, internal workflow/task steps, custom user types —
# defaults to DEBUG.
#
# Hard-coded as strings (rather than importing `SpanType`) so this module can
# be imported by `mlflow.entities.span` itself without creating a cycle. The
# `SpanType` constants are stable wire values; keep this set in sync with them.
_INFO_SPAN_TYPES: frozenset[str] = frozenset({
    "LLM",
    "CHAT_MODEL",
    "TOOL",
    "RETRIEVER",
    "AGENT",
    "EMBEDDING",
})


def default_log_level_for_span_type(span_type: str | None) -> SpanLogLevel:
    """
    Return the default :class:`SpanLogLevel` for a span of the given type.
    Used by the :class:`LiveSpan` constructor so every span — autologged or
    manual — gets a sensible level for the UI verbosity filter without users
    annotating every call.
    """
    if span_type in _INFO_SPAN_TYPES:
        return SpanLogLevel.INFO
    return SpanLogLevel.DEBUG
