import { describe, expect, test } from '@jest/globals';
import { buildFilter, buildOrderBy } from './buildTracesV4SearchParams';

describe('buildFilter', () => {
  test('returns undefined when there are no clauses', () => {
    expect(buildFilter({})).toBeUndefined();
    expect(buildFilter({ searchQuery: '   ' })).toBeUndefined();
  });

  test('a trace-id search uses the indexed request_id equality lookup (lowercased)', () => {
    const traceId = 'tr-0123456789ABCDEF0123456789abcdef';
    expect(buildFilter({ searchQuery: traceId })).toBe(`attributes.request_id = '${traceId.toLowerCase()}'`);
  });

  test('a bare 32-hex id is prefixed and lowercased into the request_id lookup', () => {
    expect(buildFilter({ searchQuery: '0123456789ABCDEF0123456789abcdef' })).toBe(
      "attributes.request_id = 'tr-0123456789abcdef0123456789abcdef'",
    );
  });

  test('a trace:/<loc>/<id> paste resolves to the request_id lookup (location dropped)', () => {
    expect(buildFilter({ searchQuery: 'trace:/cat.sch/tr-0123456789ABCDEF0123456789abcdef' })).toBe(
      "attributes.request_id = 'tr-0123456789abcdef0123456789abcdef'",
    );
  });

  test('a trace:/<loc>/<barehex> paste is also normalized to the request_id lookup', () => {
    expect(buildFilter({ searchQuery: 'trace:/cat.sch/0123456789ABCDEF0123456789abcdef' })).toBe(
      "attributes.request_id = 'tr-0123456789abcdef0123456789abcdef'",
    );
  });

  test('a trace-id lookup ignores the time range and other clauses (found regardless of window)', () => {
    expect(
      buildFilter({
        searchQuery: '0123456789abcdef0123456789abcdef',
        timeRange: { startTime: '1000', endTime: '2000' },
        extraClauses: ["attributes.status = 'ERROR'"],
      }),
    ).toBe("attributes.request_id = 'tr-0123456789abcdef0123456789abcdef'");
  });

  test('a non-trace-id search uses ILIKE on trace.text (span content)', () => {
    expect(buildFilter({ searchQuery: 'refund policy' })).toBe("trace.text ILIKE '%refund policy%'");
  });

  test('a single quote in the free-text query is escaped so the literal stays well-formed', () => {
    expect(buildFilter({ searchQuery: "O'Reilly" })).toBe("trace.text ILIKE '%O''Reilly%'");
  });

  test('time bounds compare attributes.timestamp_ms', () => {
    expect(buildFilter({ timeRange: { startTime: '1000', endTime: '2000' } })).toBe(
      'attributes.timestamp_ms > 1000 AND attributes.timestamp_ms < 2000',
    );
  });

  test('only-start or only-end time bounds emit a single clause', () => {
    expect(buildFilter({ timeRange: { startTime: '1000' } })).toBe('attributes.timestamp_ms > 1000');
    expect(buildFilter({ timeRange: { endTime: '2000' } })).toBe('attributes.timestamp_ms < 2000');
  });

  test('combines search, time bounds, and extra clauses with AND', () => {
    const filter = buildFilter({
      searchQuery: 'hello',
      timeRange: { startTime: '1000' },
      extraClauses: ["attributes.status = 'ERROR'"],
    });
    expect(filter).toBe(
      "trace.text ILIKE '%hello%' AND attributes.timestamp_ms > 1000 AND attributes.status = 'ERROR'",
    );
  });

  test('ignores blank extra clauses', () => {
    expect(buildFilter({ extraClauses: ['', '  '] })).toBeUndefined();
  });
});

describe('buildOrderBy', () => {
  test('start_time maps to timestamp with direction', () => {
    expect(buildOrderBy('start_time', 'desc')).toEqual(['timestamp DESC']);
    expect(buildOrderBy('start_time', 'asc')).toEqual(['timestamp ASC']);
  });

  test('duration maps to execution_time with direction', () => {
    expect(buildOrderBy('duration', 'desc')).toEqual(['execution_time DESC']);
    expect(buildOrderBy('duration', 'asc')).toEqual(['execution_time ASC']);
  });

  test('non-server-sortable columns return undefined (fall back to default ordering)', () => {
    expect(buildOrderBy('trace_id', 'asc')).toBeUndefined();
    expect(buildOrderBy('tokens', 'desc')).toBeUndefined();
    expect(buildOrderBy('input', 'desc')).toBeUndefined();
    expect(buildOrderBy('cost', 'asc')).toBeUndefined();
  });
});
