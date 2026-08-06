import { SpanLogLevel, SpanType } from './constants';

/**
 * Span types whose default level is INFO. These represent the user-visible
 * semantic operations (model calls, tool calls, retrievals, agent turns, etc.)
 * that should remain visible at the default UI threshold. Everything else --
 * chain glue, output parsing, internal/custom types -- defaults to DEBUG.
 *
 * Mirrors the Python `_INFO_SPAN_TYPES` set in
 * `mlflow/tracing/utils/default_log_level.py`. Keep the two in sync.
 */
const INFO_SPAN_TYPES: ReadonlySet<string> = new Set<string>([
  SpanType.LLM,
  SpanType.CHAT_MODEL,
  SpanType.TOOL,
  SpanType.RETRIEVER,
  SpanType.AGENT,
  SpanType.EMBEDDING,
]);

/**
 * Return the default SpanLogLevel for a span of the given type. Used by the
 * `LiveSpan` constructor so every span -- autologged or manual -- gets a
 * sensible level for the trace explorer's verbosity filter without users
 * annotating every call.
 */
export function defaultLogLevelForSpanType(spanType: string | undefined | null): SpanLogLevel {
  if (spanType && INFO_SPAN_TYPES.has(spanType)) {
    return SpanLogLevel.INFO;
  }
  return SpanLogLevel.DEBUG;
}
