import { SpanStatusCode } from '@opentelemetry/api';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';

import { UCSchemaSpanProcessor, UCSchemaSpanExporter } from '../../src/exporters/uc_table';
import { InMemoryTraceManager } from '../../src/core/trace_manager';
import { TraceInfo } from '../../src/core/entities/trace_info';
import { createTraceLocationFromUCSchema } from '../../src/core/entities/trace_location';
import { TraceState } from '../../src/core/entities/trace_state';
import { SpanAttributeKey, TraceMetadataKey } from '../../src/core/constants';
import * as configModule from '../../src/core/config';
import { createTestSpan, createMockOtelSpan } from '../helper';

describe('UCSchema exporters', () => {
  const CATALOG = 'catalog';
  const SCHEMA = 'schema';
  const DATABRICKS_HOST = 'https://example.databricks.com';
  const TOKEN = 'dapi-xxxx';

  beforeEach(() => {
    jest.clearAllMocks();
    // Reset in-memory manager state
    InMemoryTraceManager.reset();

    // Provide config expected by UCSchema exporters/processors
    jest.spyOn(configModule, 'getConfig').mockReturnValue({
      trackingUri: 'databricks',
      location: { catalog_name: CATALOG, schema_name: SCHEMA },
      host: DATABRICKS_HOST,
      databricksToken: TOKEN
    } as any);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('UCSchemaSpanProcessor', () => {
    it('onStart assigns v4 trace id and registers trace', () => {
      const dummyExporter = {
        export: jest.fn(),
        shutdown: jest.fn(),
        forceFlush: jest.fn()
      } as any;
      const processor = new UCSchemaSpanProcessor(dummyExporter);

      const otelTraceId = 'test-trace-id-123';
      const mockSpan = createMockOtelSpan('root', 'span-1', otelTraceId);
      // Ensure startTime is available for requestTime computation
      mockSpan.startTime = [1, 0];

      // Cast only for the onStart call which expects an OTel span
      processor.onStart(mockSpan as unknown as any, {} as any);

      const expectedTraceId = `trace:/${CATALOG}.${SCHEMA}/${otelTraceId}`;
      // Attribute is set on the span
      expect(mockSpan.getAttribute(SpanAttributeKey.TRACE_ID)).toBe(expectedTraceId);
      // Mapping between OTel trace id and MLflow v4 trace id is registered
      expect(InMemoryTraceManager.getInstance().getMlflowTraceIdFromOtelId(otelTraceId)).toBe(
        expectedTraceId
      );
    });

    it('onEnd updates trace info, aggregates token usage, and exports only for root span', () => {
      // Use a noop exporter so the trace is not popped from InMemoryTraceManager
      const dummyExporter = {
        export: jest.fn(),
        shutdown: jest.fn(),
        forceFlush: jest.fn()
      } as any;
      const processor = new UCSchemaSpanProcessor(dummyExporter);

      const otelTraceId = 'otel-trace-xyz';
      const rootStartHrTime: [number, number] = [10, 0];

      // Start: registers trace and sets initial info
      const startSpan = createMockOtelSpan('root', 'span-root', otelTraceId) as any;
      startSpan.startTime = rootStartHrTime;
      processor.onStart(startSpan, {} as any);

      // Register a child live span with token usage to be aggregated
      const mlflowTraceId = `trace:/${CATALOG}.${SCHEMA}/${otelTraceId}`;
      const liveSpan = createTestSpan('child', mlflowTraceId, 'child-1');
      liveSpan.setAttribute(SpanAttributeKey.TOKEN_USAGE, {
        input_tokens: 3,
        output_tokens: 7,
        total_tokens: 10
      });
      InMemoryTraceManager.getInstance().registerSpan(liveSpan);

      // End root span; processor should export via underlying exporter
      const endSpan = {
        name: 'root',
        parentSpanContext: undefined,
        spanContext: () => ({ traceId: otelTraceId, spanId: 'span-root' }),
        status: { code: SpanStatusCode.OK },
        startTime: rootStartHrTime,
        endTime: [12, 0]
      } as any;

      processor.onEnd(endSpan);

      // Exporter export is called once for the root span
      expect(dummyExporter.export).toHaveBeenCalled();

      // Token usage aggregated onto trace info
      const trace = InMemoryTraceManager.getInstance().getTrace(mlflowTraceId)!;
      expect(trace.info.traceMetadata[TraceMetadataKey.TOKEN_USAGE]).toBe(
        JSON.stringify({ input_tokens: 3, output_tokens: 7, total_tokens: 10 })
      );
    });
  });

  describe('UCSchemaSpanExporter', () => {
    it('exports spans via OTLP and logs TraceInfo via createTraceV4 for root spans', async () => {
      const baseExportSpy = jest
        .spyOn(OTLPTraceExporter.prototype as any, 'export')
        .mockImplementation((_spans: any, _cb: any) => {});

      const mockClient = { createTraceV4: jest.fn().mockResolvedValue({}) } as any;
      const exporter = new UCSchemaSpanExporter(mockClient);

      // Prepare in-memory trace so popTrace returns a Trace
      const otelTraceId = 'otel-123';
      const traceInfo = new TraceInfo({
        traceId: `trace:/${CATALOG}.${SCHEMA}/${otelTraceId}`,
        traceLocation: createTraceLocationFromUCSchema(CATALOG, SCHEMA),
        requestTime: 0,
        state: TraceState.IN_PROGRESS,
        traceMetadata: {},
        tags: {},
        assessments: []
      } as any);
      InMemoryTraceManager.getInstance().registerTrace(otelTraceId, traceInfo);

      const rootSpan = {
        name: 'root',
        parentSpanContext: undefined,
        spanContext: () => ({ traceId: otelTraceId, spanId: 'span-1' })
      } as any;

      exporter.export([rootSpan], () => {});
      await exporter.forceFlush();

      expect(baseExportSpy).toHaveBeenCalled();
      expect(mockClient.createTraceV4).toHaveBeenCalledTimes(1);
      expect(mockClient.createTraceV4).toHaveBeenCalledWith(traceInfo);
    });

    it('does not log TraceInfo for child spans (still sends via OTLP)', async () => {
      const baseExportSpy = jest
        .spyOn(OTLPTraceExporter.prototype as any, 'export')
        .mockImplementation((_spans: any, _cb: any) => {});

      const mockClient = { createTraceV4: jest.fn().mockResolvedValue({}) } as any;
      const exporter = new UCSchemaSpanExporter(mockClient);

      const childSpan = {
        name: 'child',
        parentSpanContext: { spanId: 'parent-1' },
        spanContext: () => ({ traceId: 'otel-child', spanId: 'span-2' })
      } as any;

      exporter.export([childSpan], () => {});
      await exporter.forceFlush();

      expect(baseExportSpy).toHaveBeenCalled();
      expect(mockClient.createTraceV4).not.toHaveBeenCalled();
    });
  });
});
