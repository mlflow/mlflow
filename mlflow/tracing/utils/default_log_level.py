from __future__ import annotations

from mlflow.entities.span import SpanType
from mlflow.entities.span_log_level import SpanLogLevel

# Span types whose autolog default is DEBUG. These represent internal/glue work
# (chain steps, output parsing, reranking) that is rarely interesting outside of
# debugging. Every other type — including custom user-defined types — defaults
# to INFO so a user-visible operation isn't accidentally hidden.
_DEBUG_SPAN_TYPES: frozenset[str] = frozenset({
    SpanType.CHAIN,
    SpanType.PARSER,
    SpanType.RERANKER,
})


def default_log_level_for_span_type(span_type: str | None) -> SpanLogLevel:
    """
    Return the default :class:`SpanLogLevel` for an autolog-created span of the
    given type, so the UI verbosity filter has a sensible value to act on
    without users annotating every wrapped library call.
    """
    if span_type in _DEBUG_SPAN_TYPES:
        return SpanLogLevel.DEBUG
    return SpanLogLevel.INFO
