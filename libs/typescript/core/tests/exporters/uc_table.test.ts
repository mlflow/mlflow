import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { ROOT_CONTEXT, SpanKind } from '@opentelemetry/api';
import {
  BasicTracerProvider,
  ReadableSpan as OTelReadableSpan,
  Span as OTelSpan,
} from '@opentelemetry/sdk-trace-base';

// Mock the OTLP proto exporter. The real implementation does dynamic
// `import('http')` calls that Jest needs --experimental-vm-modules to run.
// We assert on constructor args (url + UC table header) and assume the
// upstream OTel library serializes correctly.
const mockExport = jest.fn();
const mockShutdown = jest.fn().mockResolvedValue(undefined);
const exporterCtors: { url?: string; headers?: Record<string, string> }[] = [];
jest.mock('@opentelemetry/exporter-trace-otlp-proto', () => ({
  OTLPTraceExporter: jest
    .fn()
    .mockImplementation((cfg: { url?: string; headers?: Record<string, string> }) => {
      exporterCtors.push(cfg);
      return {
        export: (spans: unknown[], cb: (r: { code: number }) => void) => {
          mockExport(spans);
          cb({ code: 0 });
        },
        shutdown: mockShutdown,
      };
    }),
}));

import { AuthProvider } from '../../src/auth';
import { MlflowClient } from '../../src/clients/client';
import {
  TraceLocationType,
  createTraceLocationFromUcTablePrefix,
} from '../../src/core/entities/trace_location';
import { TraceInfo } from '../../src/core/entities/trace_info';
import { TraceState } from '../../src/core/entities/trace_state';
import { DATABRICKS_UC_TABLE_HEADER, TraceMetadataKey } from '../../src/core/constants';

const testHost = 'https://dbc-12345.cloud.databricks.com';
const testToken = 'test-token';

const mockAuthProvider: AuthProvider = {
  getHost: () => testHost,
  getHeadersProvider: () => () =>
    Promise.resolve({
      'Content-Type': 'application/json',
      Authorization: `Bearer ${testToken}`,
    }),
  getDatabricksToken: () => testToken,
};

describe('MlflowClient UC methods', () => {
  let server: ReturnType<typeof setupServer>;
  let client: MlflowClient;

  beforeAll(() => {
    server = setupServer();
    server.listen();
  });

  afterAll(() => {
    server.close();
  });

  beforeEach(() => {
    client = new MlflowClient({ trackingUri: 'databricks', authProvider: mockAuthProvider });
  });

  afterEach(() => {
    server.resetHandlers();
  });

  it('createTraceInfoV4 POSTs trace_info to the V4 endpoint with tags preserved', async () => {
    const location = 'cat.sch.tbl';
    const otelTraceId = 'abcdef1234567890abcdef1234567890';
    const traceInfo = new TraceInfo({
      traceId: `trace:/${location}/${otelTraceId}`,
      traceLocation: createTraceLocationFromUcTablePrefix('cat', 'sch', 'tbl'),
      requestTime: 1_700_000_000_000,
      executionDuration: 1500,
      state: TraceState.OK,
      tags: { user_id: 'u1', family_id: 'f1', conversation_id: 'c1' },
      traceMetadata: { [TraceMetadataKey.SCHEMA_VERSION]: '4' },
    });

    let capturedBody: unknown = null;
    let capturedUrl: string | null = null;
    server.use(
      http.post(
        `${testHost}/api/4.0/mlflow/traces/${encodeURIComponent(location)}/${otelTraceId}/info`,
        async ({ request }) => {
          capturedUrl = request.url;
          capturedBody = await request.json();
          // Echo back a TraceInfo with the backend populating the spans table.
          return HttpResponse.json({
            trace_id: traceInfo.traceId,
            trace_location: {
              type: TraceLocationType.UC_TABLE_PREFIX,
              uc_table_prefix: {
                catalog_name: 'cat',
                schema_name: 'sch',
                table_prefix: 'tbl',
                otel_spans_table_name: 'cat.sch.tbl_otel_spans',
              },
            },
            request_time: new Date(traceInfo.requestTime).toISOString(),
            execution_duration: '1.5s',
            state: TraceState.OK,
            trace_metadata: traceInfo.traceMetadata,
            tags: traceInfo.tags,
            assessments: [],
          });
        },
      ),
    );

    const returned = await client.createTraceInfoV4(location, otelTraceId, traceInfo);

    expect(capturedUrl).toContain(
      `/api/4.0/mlflow/traces/${encodeURIComponent(location)}/${otelTraceId}/info`,
    );
    // Body is the TraceInfo JSON directly (Databricks RPC convention).
    expect(capturedBody).toMatchObject({
      trace_id: traceInfo.traceId,
      tags: { user_id: 'u1', family_id: 'f1', conversation_id: 'c1' },
      trace_metadata: { [TraceMetadataKey.SCHEMA_VERSION]: '4' },
      trace_location: {
        type: TraceLocationType.UC_TABLE_PREFIX,
        uc_table_prefix: {
          catalog_name: 'cat',
          schema_name: 'sch',
          table_prefix: 'tbl',
        },
      },
    });
    expect(returned.traceLocation.ucTablePrefix?.otelSpansTableName).toBe('cat.sch.tbl_otel_spans');
  });

  it('exportOtlpSpansToUc constructs an OTLP proto exporter with the UC table header', async () => {
    exporterCtors.length = 0;
    mockExport.mockClear();

    const provider = new BasicTracerProvider();
    const tracer = provider.getTracer('uc-test');
    const span = tracer.startSpan('root', { kind: SpanKind.INTERNAL }, ROOT_CONTEXT) as OTelSpan;
    span.setAttribute('user.id', 'u1');
    span.end();

    await client.exportOtlpSpansToUc(
      [span as unknown as OTelReadableSpan],
      'cat.sch.tbl_otel_spans',
    );

    expect(exporterCtors).toHaveLength(1);
    expect(exporterCtors[0].url).toBe(`${testHost}/api/2.0/otel/v1/traces`);
    expect(exporterCtors[0].headers?.[DATABRICKS_UC_TABLE_HEADER]).toBe('cat.sch.tbl_otel_spans');
    expect(exporterCtors[0].headers?.['Authorization']).toBe(`Bearer ${testToken}`);
    expect(mockExport).toHaveBeenCalledTimes(1);
    expect((mockExport.mock.calls[0][0] as unknown[])[0]).toBe(span);
  });

  it('exportOtlpSpansToUc short-circuits with no spans', async () => {
    // No mock registered: if it tried to hit the network the call would throw.
    await expect(client.exportOtlpSpansToUc([], 'cat.sch.tbl_otel_spans')).resolves.toBeUndefined();
  });
});
