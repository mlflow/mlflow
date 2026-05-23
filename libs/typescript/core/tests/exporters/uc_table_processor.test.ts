import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { context, SpanKind, ROOT_CONTEXT } from '@opentelemetry/api';
import {
  BasicTracerProvider,
  ReadableSpan as OTelReadableSpan,
  Span as OTelSpan,
} from '@opentelemetry/sdk-trace-base';

// Mock the OTLP proto exporter (Jest can't load its dynamic http imports).
const exporterCtors: { url?: string; headers?: Record<string, string> }[] = [];
jest.mock('@opentelemetry/exporter-trace-otlp-proto', () => ({
  OTLPTraceExporter: jest.fn().mockImplementation((cfg: { url?: string; headers?: Record<string, string> }) => {
    exporterCtors.push(cfg);
    return {
      export: (_spans: unknown[], cb: (r: { code: number }) => void) => cb({ code: 0 }),
      shutdown: () => Promise.resolve(),
    };
  }),
}));

import { AuthProvider } from '../../src/auth';
import { MlflowClient } from '../../src/clients/client';
import {
  DatabricksUCTableSpanExporter,
  DatabricksUCTableSpanProcessor,
} from '../../src/exporters/uc_table';
import { unityCatalogDestination } from '../../src/core/destination';
import {
  TRACE_SCHEMA_VERSION_V4,
  TraceMetadataKey,
  DATABRICKS_UC_TABLE_HEADER,
} from '../../src/core/constants';
import { InMemoryTraceManager } from '../../src/core/trace_manager';
import {
  TraceLocationType,
  isUcTraceLocation,
} from '../../src/core/entities/trace_location';

const testHost = 'https://dbc-12345.cloud.databricks.com';

const mockAuthProvider: AuthProvider = {
  getHost: () => testHost,
  // eslint-disable-next-line require-await, @typescript-eslint/require-await
  getHeadersProvider: () => async () => ({
    'Content-Type': 'application/json',
    Authorization: 'Bearer test',
  }),
  getDatabricksToken: () => 'test',
};

function makeOtelRootSpan(): OTelSpan {
  // Use a real tracer to get a real Span object whose context() works.
  const provider = new BasicTracerProvider();
  const tracer = provider.getTracer('uc-test');
  const span = tracer.startSpan('root', { kind: SpanKind.INTERNAL }, ROOT_CONTEXT) as OTelSpan;
  return span;
}

describe('DatabricksUCTableSpanProcessor + Exporter end-to-end', () => {
  let server: ReturnType<typeof setupServer>;
  let traceInfoCalls: { url: string; body: any }[];

  beforeAll(() => {
    server = setupServer();
    server.listen();
  });

  afterAll(() => {
    server.close();
  });

  beforeEach(() => {
    traceInfoCalls = [];
    exporterCtors.length = 0;
    server.resetHandlers();
    server.use(
      http.post(
        `${testHost}/api/4.0/mlflow/traces/:location/:otelTraceId/info`,
        async ({ request, params }) => {
          const body = (await request.json()) as any;
          traceInfoCalls.push({ url: request.url, body });
          return HttpResponse.json({
            trace_id: body.trace_id,
            trace_location: {
              type: TraceLocationType.UC_TABLE_PREFIX,
              uc_table_prefix: {
                catalog_name: 'cat',
                schema_name: 'sch',
                table_prefix: 'tbl',
                otel_spans_table_name: 'cat.sch.tbl_otel_spans',
              },
            },
            request_time: body.request_time,
            execution_duration: body.execution_duration,
            state: body.state,
            trace_metadata: body.trace_metadata,
            tags: body.tags,
            assessments: [],
          });
          void params;
        },
      ),
    );
  });

  it('persists trace tags via V4 endpoint and uploads spans via OTLP with UC header', async () => {
    const client = new MlflowClient({
      trackingUri: 'databricks',
      authProvider: mockAuthProvider,
    });
    const dest = unityCatalogDestination({
      catalogName: 'cat',
      schemaName: 'sch',
      tablePrefix: 'tbl',
    });
    const exporter = new DatabricksUCTableSpanExporter(client);
    const processor = new DatabricksUCTableSpanProcessor(exporter, dest);

    const span = makeOtelRootSpan();
    processor.onStart(span, context.active());

    // Simulate `updateCurrentTrace({ tags: ... })`: the user mutates the
    // in-memory TraceInfo recorded by the processor in onStart.
    const otelTraceId = span.spanContext().traceId;
    const mgr = InMemoryTraceManager.getInstance();
    const mlflowTraceId = mgr.getMlflowTraceIdFromOtelId(otelTraceId)!;
    expect(mlflowTraceId.startsWith('trace:/cat.sch.tbl/')).toBe(true);
    const trace = mgr.getTrace(mlflowTraceId)!;
    Object.assign(trace.info.tags, {
      user_id: 'u1',
      family_id: 'f1',
      conversation_id: 'c1',
    });

    expect(isUcTraceLocation(trace.info.traceLocation)).toBe(true);
    expect(trace.info.traceMetadata[TraceMetadataKey.SCHEMA_VERSION]).toBe(TRACE_SCHEMA_VERSION_V4);

    span.end();
    processor.onEnd(span as unknown as OTelReadableSpan);
    await processor.forceFlush();

    expect(traceInfoCalls).toHaveLength(1);
    // The Databricks RPC convention sends the TraceInfo JSON directly as the
    // body (not wrapped in `{ trace_info: ... }`), so the body IS the trace info.
    const posted = traceInfoCalls[0].body;
    expect(posted.trace_id).toBe(mlflowTraceId);
    expect(posted.tags).toEqual({ user_id: 'u1', family_id: 'f1', conversation_id: 'c1' });
    expect(posted.trace_metadata[TraceMetadataKey.SCHEMA_VERSION]).toBe(TRACE_SCHEMA_VERSION_V4);
    expect(posted.trace_location.uc_table_prefix).toEqual({
      catalog_name: 'cat',
      schema_name: 'sch',
      table_prefix: 'tbl',
    });

    expect(exporterCtors).toHaveLength(1);
    expect(exporterCtors[0].url).toBe(`${testHost}/api/2.0/otel/v1/traces`);
    expect(exporterCtors[0].headers?.[DATABRICKS_UC_TABLE_HEADER]).toBe(
      'cat.sch.tbl_otel_spans',
    );
  });
});
