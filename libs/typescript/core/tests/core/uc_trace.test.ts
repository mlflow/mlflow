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
  setDestination,
  getDestination,
  resetDestination,
  unityCatalogDestination,
  ucSchemaDestination,
} from '../../src/core/destination';
import { TraceInfo } from '../../src/core/entities/trace_info';
import { TraceState } from '../../src/core/entities/trace_state';
import { spansToOtlpRequest } from '../../src/exporters/otlp';

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

describe('OTLP serialization', () => {
  it('returns an empty resourceSpans for no input', () => {
    expect(spansToOtlpRequest([])).toEqual({ resourceSpans: [] });
  });

  it('encodes a single span with attributes and status', () => {
    const fakeSpan = {
      spanContext: () => ({ traceId: 'aabb', spanId: 'ccdd', traceFlags: 1, isRemote: false }),
      parentSpanContext: undefined,
      name: 'root',
      kind: 1,
      startTime: [1, 500] as [number, number],
      endTime: [2, 0] as [number, number],
      attributes: { 'user.id': 'u1', 'mlflow.spanType': 'CHAIN' },
      events: [],
      status: { code: 1, message: undefined },
      resource: { attributes: { 'service.name': 'svc' } },
      instrumentationScope: { name: 'mlflow-tracing' },
    } as unknown as Parameters<typeof spansToOtlpRequest>[0][number];

    const req = spansToOtlpRequest([fakeSpan]);
    expect(req.resourceSpans).toHaveLength(1);
    expect(req.resourceSpans[0].scopeSpans).toHaveLength(1);
    const otlpSpan = req.resourceSpans[0].scopeSpans[0].spans[0];
    expect(otlpSpan.traceId).toBe('aabb');
    expect(otlpSpan.name).toBe('root');
    expect(otlpSpan.startTimeUnixNano).toBe('1000000500');
    expect(otlpSpan.endTimeUnixNano).toBe('2000000000');
    expect(otlpSpan.status.code).toBe(1);
    const attrMap = Object.fromEntries(
      otlpSpan.attributes.map((kv) => [kv.key, kv.value.stringValue]),
    );
    expect(attrMap['user.id']).toBe('u1');
    expect(attrMap['mlflow.spanType']).toBe('CHAIN');
  });
});
