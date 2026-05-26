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

describe('init() with Databricks tracking URI auto-resolves UC trace location', () => {
  let server: ReturnType<typeof setupServer>;
  let getExperimentCalls: string[];
  let v4TraceInfoCalls: { url: string; body: any }[];

  beforeAll(() => {
    process.env.DATABRICKS_HOST = testHost;
    process.env.DATABRICKS_TOKEN = 'test-token';
    server = setupServer();
    server.listen();
  });

  afterAll(() => {
    server.close();
    delete process.env.DATABRICKS_HOST;
    delete process.env.DATABRICKS_TOKEN;
  });

  beforeEach(() => {
    getExperimentCalls = [];
    v4TraceInfoCalls = [];
    exporterCtors.length = 0;
    server.resetHandlers();
    // Catch-all V3 + V4 + OTLP handlers so deferred exports from previous
    // tests don't hit unmatched-request warnings on this test's reset state.
    server.use(
      http.post(`${testHost}/api/3.0/mlflow/traces`, async ({ request }) => {
        const body = (await request.json()) as any;
        return HttpResponse.json({ trace: body.trace });
      }),
      http.post(
        `${testHost}/api/4.0/mlflow/traces/:location/:otelTraceId/info`,
        async ({ request }) => {
          const body = (await request.json()) as any;
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
  });

  it('wires the UC processor when the experiment carries Databricks UC trace tags', async () => {
    server.use(
      http.get(`${testHost}/api/2.0/mlflow/experiments/get`, ({ request }) => {
        const url = new URL(request.url);
        getExperimentCalls.push(url.searchParams.get('experiment_id') ?? '');
        return HttpResponse.json({
          experiment: {
            experiment_id: '123',
            name: 'uc-exp',
            tags: [
              { key: 'mlflow.experiment.databricksTraceDestinationPath', value: 'cat.sch.prefix' },
              {
                key: 'mlflow.experiment.databricksTraceSpanStorageTable',
                value: 'cat.sch.prefix_otel_spans',
              },
            ],
          },
        });
      }),
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

    await init({ trackingUri: 'databricks', experimentId: '123' });

    expect(getExperimentCalls).toEqual(['123']);

    void withSpan(() => {}, { name: 'root' });
    await flushTraces();

    expect(v4TraceInfoCalls).toHaveLength(1);
    expect(v4TraceInfoCalls[0].body.trace_id).toMatch(/^trace:\/cat\.sch\.prefix\//);
  });

  it('falls back to the V3 processor when the experiment has no UC trace tags', async () => {
    server.use(
      http.get(`${testHost}/api/2.0/mlflow/experiments/get`, () =>
        HttpResponse.json({
          experiment: { experiment_id: '456', name: 'plain', tags: [] },
        }),
      ),
    );

    await init({ trackingUri: 'databricks', experimentId: '456' });

    void withSpan(() => {}, { name: 'root' });
    await flushTraces();

    // V4 endpoint must not be hit when UC isn't configured.
    expect(v4TraceInfoCalls).toHaveLength(0);
  });
});
