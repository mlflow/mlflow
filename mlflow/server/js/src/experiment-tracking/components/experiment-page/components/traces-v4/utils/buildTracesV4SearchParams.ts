import { type SortDirection, type TraceColumnId } from '@databricks/web-shared/traces-table';
import { isV4TraceId, parseV4TraceId } from '@databricks/web-shared/model-trace-explorer';

/** A canonical trace-id token, e.g. `tr-1234…`. Matched case-insensitively for the indexed fast path. */
const TRACE_ID_PATTERN = /^tr-[0-9a-f]{32}$/i;
/** A bare 32-hex trace id (no `tr-` prefix), mirroring v3's `HEX_TRACE_ID_PATTERN`. */
const BARE_HEX_TRACE_ID_PATTERN = /^[0-9a-f]{32}$/i;

/**
 * Recognize the three trace-id search formats and return the canonical stored form
 * (`tr-<lowerhex>`), else `undefined`:
 * - `tr-<32hex>` → lowercased.
 * - bare `<32hex>` → `tr-` prefixed, lowercased.
 * - `trace:/<loc>/<id>` → the extracted id (already `tr-…` or bare), normalized as above.
 *
 * Stored trace ids are normalized to the `tr-`-prefixed lowercase form (`StartTraceV3Handler.scala`),
 * so matching this against the indexed `attributes.request_id` column is the fast path.
 */
const normalizeTraceIdQuery = (query: string): string | undefined => {
  if (TRACE_ID_PATTERN.test(query)) {
    return query.toLowerCase();
  }
  if (BARE_HEX_TRACE_ID_PATTERN.test(query)) {
    return `tr-${query.toLowerCase()}`;
  }
  // A full `trace:/<loc>/<id>` paste: the location is intentionally dropped (the tab already scopes
  // to its single UC location), so only the extracted id is normalized.
  if (isV4TraceId(query)) {
    const traceId = parseV4TraceId(query)?.trace_id;
    return traceId ? normalizeTraceIdQuery(traceId) : undefined;
  }
  return undefined;
};

/**
 * Escape a user-supplied value for interpolation inside a single-quoted search-filter literal.
 * The search parser treats `'` as the literal delimiter, so a value like `O'Reilly` would close the
 * literal early and yield invalid syntax the backend rejects; doubling the quote (`''`) is the
 * standard SQL-string escape the parser understands. Every clause that wraps a value in `'…'` must
 * route it through here.
 */
export const escapeFilterValue = (value: string): string => value.replace(/'/g, "''");

export interface BuildFilterParams {
  /** Free-text search. A trace-id token becomes an indexed `request_id` lookup; otherwise ILIKE. */
  searchQuery?: string;
  /** Millisecond epoch bounds (strings, as produced by `useMonitoringFiltersTimeRange`). */
  timeRange?: { startTime?: string; endTime?: string };
  /** Extra filter clauses from the Filter popover, already rendered as `field OP value` strings. */
  extraClauses?: string[];
}

/**
 * Build the `filter` string for `ajax-api/4.0/mlflow/traces/search`.
 *
 * Siloed from the shared `createMlflowSearchFilter` so the V4 tab owns exactly the clauses it
 * emits, but it mirrors the shared behavior for the pieces it shares: a trace-id search uses the
 * indexed `attributes.request_id` equality lookup (see mlflow discussion #21193) instead of a
 * full content scan, and time bounds compare `attributes.timestamp_ms`. A trace-id search returns
 * that lone clause (no time/other clauses) so the exact trace is found regardless of the range.
 *
 * Returns `undefined` (not an empty string) when there are no clauses, matching what the search
 * API expects for "no filter".
 */
export const buildFilter = ({ searchQuery, timeRange, extraClauses }: BuildFilterParams): string | undefined => {
  const trimmedQuery = searchQuery?.trim();

  // A recognized trace-id (any of the three formats) resolves via the indexed `request_id` equality
  // lookup, and — like v3's direct-fetch path — it deliberately ignores the time range and every
  // other filter clause: pasting a trace id should surface that exact trace regardless of the
  // selected window (otherwise the default short window hides older ids). Return early with just the
  // id clause so no `timestamp_ms`/popover clause narrows it away.
  if (trimmedQuery) {
    const normalizedTraceId = normalizeTraceIdQuery(trimmedQuery);
    if (normalizedTraceId) {
      return `attributes.request_id = '${normalizedTraceId}'`;
    }
  }

  const clauses: string[] = [];

  if (trimmedQuery) {
    // Free-text search is a content substring scan. OSS's search parser exposes trace content under
    // the `text` search key (aliased to span content), NOT `request` — `trace.request` is rejected as
    // an invalid attribute key. This matches V3's `createMlflowSearchFilter` (`trace.text ILIKE …`).
    clauses.push(`trace.text ILIKE '%${escapeFilterValue(trimmedQuery)}%'`);
  }

  if (timeRange?.startTime) {
    clauses.push(`attributes.timestamp_ms > ${timeRange.startTime}`);
  }
  if (timeRange?.endTime) {
    clauses.push(`attributes.timestamp_ms < ${timeRange.endTime}`);
  }

  if (extraClauses) {
    for (const clause of extraClauses) {
      const trimmed = clause.trim();
      if (trimmed) {
        clauses.push(trimmed);
      }
    }
  }

  return clauses.length > 0 ? clauses.join(' AND ') : undefined;
};

/**
 * Build the `order_by` array for the search API. The server only supports ordering by
 * `timestamp` (start time) and `execution_time` (duration); every other column returns
 * `undefined` so the caller falls back to the server's default ordering rather than emitting an
 * `order_by` the backend would reject.
 */
export const buildOrderBy = (sort: TraceColumnId, dir: SortDirection): string[] | undefined => {
  const direction = dir === 'asc' ? 'ASC' : 'DESC';
  switch (sort) {
    case 'start_time':
      return [`timestamp ${direction}`];
    case 'duration':
      return [`execution_time ${direction}`];
    default:
      return undefined;
  }
};
