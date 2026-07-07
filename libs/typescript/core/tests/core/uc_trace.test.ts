import {
  TraceLocationType,
  createTraceLocationFromUcTablePrefix,
  getOtelSpansTableName,
  getUcLocationString,
  isUcTraceLocation,
  ucTablePrefixLocationString,
} from '../../src/core/entities/trace_location';
import {
  constructTraceIdV4,
  generateTraceIdV3,
  parseTraceIdV4,
} from '../../src/core/utils/trace_id';
import {
  DATABRICKS_TRACE_ANNOTATIONS_TABLE_TAG,
  DATABRICKS_TRACE_DESTINATION_PATH_TAG,
  DATABRICKS_TRACE_LOG_STORAGE_TABLE_TAG,
  DATABRICKS_TRACE_SPAN_STORAGE_TABLE_TAG,
  ucLocationFromExperimentTags,
} from '../../src/core/destination';
import { TraceInfo } from '../../src/core/entities/trace_info';
import { TraceState } from '../../src/core/entities/trace_state';

describe('Trace ID v4 helpers', () => {
  it('parses a v4 trace ID into location and otel trace ID', () => {
    const [location, otelId] = parseTraceIdV4(
      'trace:/cat.sch.tbl/abcdef1234567890abcdef1234567890',
    );
    expect(location).toBe('cat.sch.tbl');
    expect(otelId).toBe('abcdef1234567890abcdef1234567890');
  });

  it('returns [null, raw] for v3 trace IDs', () => {
    const [location, otelId] = parseTraceIdV4('tr-abcdef1234567890');
    expect(location).toBeNull();
    expect(otelId).toBe('tr-abcdef1234567890');
  });

  it('throws on a malformed v4 trace ID', () => {
    expect(() => parseTraceIdV4('trace:/onlyone')).toThrow(/Invalid trace ID format/);
    expect(() => parseTraceIdV4('trace://abc')).toThrow(/Invalid trace ID format/);
  });

  it('constructs v4 and v3 IDs in their canonical formats', () => {
    expect(constructTraceIdV4('cat.sch.tbl', 'deadbeef')).toBe('trace:/cat.sch.tbl/deadbeef');
    expect(generateTraceIdV3('deadbeef')).toBe('tr-deadbeef');
  });
});

describe('UC trace location helpers', () => {
  it('builds a UC table-prefix TraceLocation and returns its location string', () => {
    const loc = createTraceLocationFromUcTablePrefix('cat', 'sch', 'agent');
    expect(loc.type).toBe(TraceLocationType.UC_TABLE_PREFIX);
    expect(isUcTraceLocation(loc)).toBe(true);
    expect(getUcLocationString(loc)).toBe('cat.sch.agent');
    expect(ucTablePrefixLocationString(loc.ucTablePrefix!)).toBe('cat.sch.agent');
    // Default spans table convention: `<prefix>_otel_spans` under the same
    // catalog/schema. Customers can override by setting otelSpansTableName.
    expect(getOtelSpansTableName(loc)).toBe('cat.sch.agent_otel_spans');
  });

  it('respects an explicit backend-populated spans table name', () => {
    const loc = createTraceLocationFromUcTablePrefix('cat', 'sch', 'agent');
    loc.ucTablePrefix!.otelSpansTableName = 'cat.sch.custom_spans';
    expect(getOtelSpansTableName(loc)).toBe('cat.sch.custom_spans');
  });
});

describe('ucLocationFromExperimentTags', () => {
  it('returns null when the experiment has no Databricks trace tags', () => {
    expect(ucLocationFromExperimentTags({})).toBeNull();
    expect(ucLocationFromExperimentTags({ unrelated: 'x' })).toBeNull();
  });

  it('returns null when the destination path is not three-segment', () => {
    expect(
      ucLocationFromExperimentTags({
        [DATABRICKS_TRACE_DESTINATION_PATH_TAG]: 'cat.sch',
      }),
    ).toBeNull();
  });

  it('returns null when any path segment is empty', () => {
    expect(
      ucLocationFromExperimentTags({
        [DATABRICKS_TRACE_DESTINATION_PATH_TAG]: 'cat.sch.',
      }),
    ).toBeNull();
    expect(
      ucLocationFromExperimentTags({
        [DATABRICKS_TRACE_DESTINATION_PATH_TAG]: '.sch.prefix',
      }),
    ).toBeNull();
  });

  it('parses a UC table-prefix location and copies the spans / logs / annotations tables', () => {
    const loc = ucLocationFromExperimentTags({
      [DATABRICKS_TRACE_DESTINATION_PATH_TAG]: 'cat.sch.prefix',
      [DATABRICKS_TRACE_SPAN_STORAGE_TABLE_TAG]: 'cat.sch.prefix_otel_spans',
      [DATABRICKS_TRACE_LOG_STORAGE_TABLE_TAG]: 'cat.sch.prefix_otel_logs',
      [DATABRICKS_TRACE_ANNOTATIONS_TABLE_TAG]: 'cat.sch.prefix_annotations',
    });
    expect(loc).not.toBeNull();
    expect(loc).toEqual({
      catalogName: 'cat',
      schemaName: 'sch',
      tablePrefix: 'prefix',
      otelSpansTableName: 'cat.sch.prefix_otel_spans',
      otelLogsTableName: 'cat.sch.prefix_otel_logs',
      annotationsTableName: 'cat.sch.prefix_annotations',
    });
  });
});

describe('TraceInfo serialization with UC locations', () => {
  it('serializes and deserializes a UC table-prefix TraceLocation', () => {
    const info = new TraceInfo({
      traceId: 'trace:/cat.sch.tbl/abc',
      traceLocation: createTraceLocationFromUcTablePrefix('cat', 'sch', 'tbl'),
      requestTime: 1700000000000,
      state: TraceState.OK,
      tags: { user_id: 'u1', family_id: 'f1' },
      traceMetadata: {},
    });
    const json = info.toJson();
    expect(json.trace_location.type).toBe(TraceLocationType.UC_TABLE_PREFIX);
    expect(json.trace_location.uc_table_prefix).toEqual({
      catalog_name: 'cat',
      schema_name: 'sch',
      table_prefix: 'tbl',
    });
    expect(json.tags).toEqual({ user_id: 'u1', family_id: 'f1' });

    const roundTripped = TraceInfo.fromJson(json);
    expect(roundTripped.traceLocation.type).toBe(TraceLocationType.UC_TABLE_PREFIX);
    expect(roundTripped.traceLocation.ucTablePrefix?.catalogName).toBe('cat');
    expect(roundTripped.traceLocation.ucTablePrefix?.schemaName).toBe('sch');
    expect(roundTripped.traceLocation.ucTablePrefix?.tablePrefix).toBe('tbl');
    expect(roundTripped.tags).toEqual({ user_id: 'u1', family_id: 'f1' });
  });
});
