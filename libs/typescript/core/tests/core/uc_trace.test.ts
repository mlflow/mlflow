import {
  TraceLocationType,
  createTraceLocationFromUcSchema,
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
  destinationFromExperimentTags,
  setDestination,
  getDestination,
  resetDestination,
  unityCatalogDestination,
  ucSchemaDestination,
} from '../../src/core/destination';
import { TraceInfo } from '../../src/core/entities/trace_info';
import { TraceState } from '../../src/core/entities/trace_state';

describe('Trace ID v4 helpers', () => {
  it('parses a v4 trace ID into location and otel trace ID', () => {
    const [location, otelId] = parseTraceIdV4('trace:/cat.sch.tbl/abcdef1234567890abcdef1234567890');
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
    expect(constructTraceIdV4('cat.sch', 'deadbeef')).toBe('trace:/cat.sch/deadbeef');
    expect(generateTraceIdV3('deadbeef')).toBe('tr-deadbeef');
  });
});

describe('UC trace location helpers', () => {
  it('builds a UC schema TraceLocation and returns its location string', () => {
    const loc = createTraceLocationFromUcSchema('cat', 'sch');
    expect(loc.type).toBe(TraceLocationType.UC_SCHEMA);
    expect(isUcTraceLocation(loc)).toBe(true);
    expect(getUcLocationString(loc)).toBe('cat.sch');
    // Default UC schema spans table name is appended.
    expect(getOtelSpansTableName(loc)).toBe('cat.sch.mlflow_experiment_trace_otel_spans');
  });

  it('builds a UC table-prefix TraceLocation and returns its location string', () => {
    const loc = createTraceLocationFromUcTablePrefix('cat', 'sch', 'agent');
    expect(loc.type).toBe(TraceLocationType.UC_TABLE_PREFIX);
    expect(isUcTraceLocation(loc)).toBe(true);
    expect(getUcLocationString(loc)).toBe('cat.sch.agent');
    expect(ucTablePrefixLocationString(loc.ucTablePrefix!)).toBe('cat.sch.agent');
    // Without a backend-populated spans table the helper returns null - the
    // exporter will then skip OTLP span upload but still persist trace info.
    expect(getOtelSpansTableName(loc)).toBeNull();
  });
});

describe('setDestination / getDestination', () => {
  afterEach(() => resetDestination());

  it('round-trips a Unity Catalog destination', () => {
    const dest = unityCatalogDestination({
      catalogName: 'cat',
      schemaName: 'sch',
      tablePrefix: 'agent',
    });
    setDestination(dest);
    expect(getDestination()).toBe(dest);
  });

  it('round-trips a UC schema destination', () => {
    const dest = ucSchemaDestination({ catalogName: 'cat', schemaName: 'sch' });
    setDestination(dest);
    expect(getDestination()).toBe(dest);
  });

  it('rejects empty UC fields', () => {
    expect(() => unityCatalogDestination({ catalogName: '', schemaName: 's', tablePrefix: 't' }))
      .toThrow(/catalogName/);
    expect(() => ucSchemaDestination({ catalogName: 'c', schemaName: '' })).toThrow(/schemaName/);
  });
});

describe('destinationFromExperimentTags', () => {
  it('returns null when the experiment has no Databricks trace tags', () => {
    expect(destinationFromExperimentTags({})).toBeNull();
    expect(destinationFromExperimentTags({ unrelated: 'x' })).toBeNull();
  });

  it('returns null when the destination path is not three-segment', () => {
    expect(
      destinationFromExperimentTags({
        [DATABRICKS_TRACE_DESTINATION_PATH_TAG]: 'cat.sch',
      }),
    ).toBeNull();
  });

  it('parses a UC table-prefix destination and copies the spans / logs / annotations tables', () => {
    const dest = destinationFromExperimentTags({
      [DATABRICKS_TRACE_DESTINATION_PATH_TAG]: 'cat.sch.prefix',
      [DATABRICKS_TRACE_SPAN_STORAGE_TABLE_TAG]: 'cat.sch.prefix_otel_spans',
      [DATABRICKS_TRACE_LOG_STORAGE_TABLE_TAG]: 'cat.sch.prefix_otel_logs',
      [DATABRICKS_TRACE_ANNOTATIONS_TABLE_TAG]: 'cat.sch.prefix_annotations',
    });
    expect(dest).not.toBeNull();
    expect(dest!.kind).toBe('uc_table_prefix');
    expect(dest!.location).toEqual({
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

  it('serializes and deserializes a UC schema TraceLocation', () => {
    const info = new TraceInfo({
      traceId: 'trace:/cat.sch/abc',
      traceLocation: createTraceLocationFromUcSchema('cat', 'sch'),
      requestTime: 1700000000000,
      state: TraceState.OK,
    });
    const json = info.toJson();
    expect(json.trace_location.type).toBe(TraceLocationType.UC_SCHEMA);
    expect(json.trace_location.uc_schema).toEqual({ catalog_name: 'cat', schema_name: 'sch' });

    const roundTripped = TraceInfo.fromJson(json);
    expect(roundTripped.traceLocation.ucSchema?.catalogName).toBe('cat');
    expect(roundTripped.traceLocation.ucSchema?.schemaName).toBe('sch');
  });
});

