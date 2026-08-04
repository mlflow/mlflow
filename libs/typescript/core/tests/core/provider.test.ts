import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';

// Mock the OTLP proto exporter (Jest can't load its dynamic http imports).
const exporterCtors: { url?: string; headers?: Record<string, string> }[] = [];
jest.mock('@opentelemetry/exporter-trace-otlp-proto', () => ({
  OTLPTraceExporter: jest
    .fn()
    .mockImplementation((cfg: { url?: string; headers?: Record<string, string> }) => {
      exporterCtors.push(cfg);
      return {
        export: (_spans: unknown[], cb: (r: { code: number }) => void) => cb({ code: 0 }),
        shutdown: () => Promise.resolve(),
        forceFlush: () => Promise.resolve(),
      };
    }),
}));

import { init } from '../../src/core/config';
import { flushTraces } from '../../src/core/provider';
import { withSpan } from '../../src/core/api';

const testHost = 'https://dbc-12345.cloud.databricks.com';

// `init()` is documented as call-once-per-process: NodeSDK's background
// shutdown can race with a subsequent start and clobber the global tracer
// provider. This file therefore only exercises a single `init()` call.
// UC processor + exporter behavior is covered in detail by the unit tests
// in `tests/exporters/uc_table*.test.ts`; this file just verifies that
// `init()` wires up the UC processor when `traceLocation` is provided.
describe('init() with traceLocation wires the UC span processor', () => {
  let server: ReturnType<typeof setupServer>;
  const v4TraceInfoCalls: { url: string; body: any }[] = [];

  beforeAll(() => {
    process.env.DATABRICKS_HOST = testHost;
    process.env.DATABRICKS_TOKEN = 'test-token';
    server = setupServer(
      http.post(
        `${testHost}/api/4.0/mlflow/traces/:location/:otelTraceId/info`,
        async ({ request }) => {
          const body = (await request.json()) as any;
          v4TraceInfoCalls.push({ url: request.url, body });
          return HttpResponse.json({
            trace_id: body.trace_id,
            trace_location: body.trace_location,
            request_time: body.request_time,
            execution_duration: body.execution_duration,
            state: body.state,
            trace_metadata: body.trace_metadata,
            tags: body.tags,
            assessments: [],
          });
        },
      ),
    );
    server.listen();

    init({
      trackingUri: 'databricks',
      experimentId: '4118495900667593',
      traceLocation: { catalogName: 'cat', schemaName: 'sch', tablePrefix: 'agent' },
    });
  });

  afterAll(() => {
    server.close();
    delete process.env.DATABRICKS_HOST;
    delete process.env.DATABRICKS_TOKEN;
  });

  it('routes spans through the V4 endpoint with the configured UC location', async () => {
    void withSpan(() => {}, { name: 'root' });
    await flushTraces();

    expect(v4TraceInfoCalls).toHaveLength(1);
    expect(v4TraceInfoCalls[0].body.trace_id).toMatch(/^trace:\/cat\.sch\.agent\//);
    expect(v4TraceInfoCalls[0].body.trace_location.uc_table_prefix).toEqual({
      catalog_name: 'cat',
      schema_name: 'sch',
      table_prefix: 'agent',
    });
  });
});
