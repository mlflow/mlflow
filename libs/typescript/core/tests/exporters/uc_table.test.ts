import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { AuthProvider } from '../../src/auth';
import { MlflowClient } from '../../src/clients/client';
import {
  TraceLocationType,
  createTraceLocationFromUcTablePrefix,
} from '../../src/core/entities/trace_location';
import { TraceInfo } from '../../src/core/entities/trace_info';
import { TraceState } from '../../src/core/entities/trace_state';
import {
  DATABRICKS_UC_TABLE_HEADER,
  TRACE_SCHEMA_VERSION_V4,
  TraceMetadataKey,
} from '../../src/core/constants';

const testHost = 'https://dbc-12345.cloud.databricks.com';
const testToken = 'test-token';

const mockAuthProvider: AuthProvider = {
  getHost: () => testHost,
  // eslint-disable-next-line require-await, @typescript-eslint/require-await
  getHeadersProvider: () => async () => ({
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
      traceMetadata: { [TraceMetadataKey.SCHEMA_VERSION]: TRACE_SCHEMA_VERSION_V4 },
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

    expect(capturedUrl).toContain(`/api/4.0/mlflow/traces/${encodeURIComponent(location)}/${otelTraceId}/info`);
    expect(capturedBody).toMatchObject({
      trace_info: {
        trace_id: traceInfo.traceId,
        tags: { user_id: 'u1', family_id: 'f1', conversation_id: 'c1' },
        trace_metadata: { [TraceMetadataKey.SCHEMA_VERSION]: TRACE_SCHEMA_VERSION_V4 },
        trace_location: {
          type: TraceLocationType.UC_TABLE_PREFIX,
          uc_table_prefix: {
            catalog_name: 'cat',
            schema_name: 'sch',
            table_prefix: 'tbl',
          },
        },
      },
    });
    expect(returned.traceLocation.ucTablePrefix?.otelSpansTableName).toBe('cat.sch.tbl_otel_spans');
  });

  it('exportOtlpSpansToUc POSTs OTLP-JSON with the UC table name header', async () => {
    let capturedHeaders: Headers | null = null;
    let capturedBody: any = null;
    server.use(
      http.post(`${testHost}/api/2.0/otel/v1/traces`, async ({ request }) => {
        capturedHeaders = request.headers;
        capturedBody = await request.json();
        return HttpResponse.json({}, { status: 200 });
      }),
    );

    const fakeSpan = {
      spanContext: () => ({ traceId: 'aabb', spanId: 'ccdd', traceFlags: 1, isRemote: false }),
      parentSpanContext: undefined,
      name: 'root',
      kind: 1,
      startTime: [1, 0],
      endTime: [2, 0],
      attributes: { 'user.id': 'u1' },
      events: [],
      status: { code: 1 },
      resource: { attributes: {} },
      instrumentationScope: { name: 'mlflow-tracing' },
    } as any;

    await client.exportOtlpSpansToUc([fakeSpan], 'cat.sch.tbl_otel_spans');

    expect(capturedHeaders).not.toBeNull();
    expect(capturedHeaders!.get(DATABRICKS_UC_TABLE_HEADER.toLowerCase())).toBe(
      'cat.sch.tbl_otel_spans',
    );
    expect(capturedBody.resourceSpans).toHaveLength(1);
    expect(capturedBody.resourceSpans[0].scopeSpans[0].spans[0].name).toBe('root');
  });

  it('exportOtlpSpansToUc short-circuits with no spans', async () => {
    // No mock registered: if it tried to hit the network the call would throw.
    await expect(client.exportOtlpSpansToUc([], 'cat.sch.tbl_otel_spans')).resolves.toBeUndefined();
  });
});
