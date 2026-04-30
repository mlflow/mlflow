import { SpanLogLevel, SpanType } from './constants';

/**
 * Span types whose autolog default is DEBUG. These represent internal/glue
 * work (chain steps, output parsing, reranking) that is rarely interesting
 * outside of debugging. Every other type -- including custom user-defined
 * types -- defaults to INFO so a user-visible operation isn't accidentally
 * hidden.
 *
 * Mirrors the Python `_DEBUG_SPAN_TYPES` set in
 * `mlflow/tracing/utils/default_log_level.py`. Keep the two in sync.
 */
const DEBUG_SPAN_TYPES: ReadonlySet<string> = new Set<string>([
  SpanType.CHAIN,
  SpanType.PARSER,
  SpanType.RERANKER,
]);

/**
 * Return the default SpanLogLevel for an autolog-created span of the given
 * type, so the trace explorer's verbosity filter has a sensible value to
 * act on without users annotating every wrapped library call.
 */
export function defaultLogLevelForSpanType(spanType: string | undefined | null): SpanLogLevel {
  if (spanType && DEBUG_SPAN_TYPES.has(spanType)) {
    return SpanLogLevel.DEBUG;
  }
  return SpanLogLevel.INFO;
}
