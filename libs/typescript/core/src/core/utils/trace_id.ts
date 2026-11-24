import { TRACE_ID_PREFIX, TRACE_ID_V4_PREFIX } from '../constants';

/**
 * Construct a V3-schema MLflow trace ID from an OTel hex trace ID.
 * Format: `tr-<hex_trace_id>`.
 */
export function generateTraceIdV3(otelHexTraceId: string): string {
  return TRACE_ID_PREFIX + otelHexTraceId;
}

/**
 * Construct a V4-schema MLflow trace ID for the given UC location and
 * OTel hex trace ID. Format: `trace:/<location>/<hex_trace_id>`.
 */
export function constructTraceIdV4(location: string, otelHexTraceId: string): string {
  return `${TRACE_ID_V4_PREFIX}${location}/${otelHexTraceId}`;
}

/**
 * Parse an MLflow trace ID. For V4 IDs returns `[location, otelHexTraceId]`;
 * for V3 IDs returns `[null, raw_id]`. Mirrors Python's `parse_trace_id_v4`.
 */
export function parseTraceIdV4(traceId: string | null | undefined): [string | null, string | null] {
  if (!traceId) {
    return [null, null];
  }
  if (traceId.startsWith(TRACE_ID_V4_PREFIX)) {
    const rest = traceId.slice(TRACE_ID_V4_PREFIX.length);
    const slash = rest.indexOf('/');
    if (slash <= 0 || slash === rest.length - 1) {
      throw new Error(
        `Invalid trace ID format: ${traceId}. ` +
          `Expected format: ${TRACE_ID_V4_PREFIX}<location>/<trace_id>`,
      );
    }
    return [rest.slice(0, slash), rest.slice(slash + 1)];
  }
  return [null, traceId];
}
