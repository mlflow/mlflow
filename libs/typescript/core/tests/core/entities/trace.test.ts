import { trace } from '@opentelemetry/api';
import { BasicTracerProvider } from '@opentelemetry/sdk-trace-node';
import { Trace } from '../../../src/core/entities/trace';
import { TraceInfo } from '../../../src/core/entities/trace_info';
import { TraceData } from '../../../src/core/entities/trace_data';
import { TraceState } from '../../../src/core/entities/trace_state';
import { createTraceLocationFromExperimentId } from '../../../src/core/entities/trace_location';
import { createMlflowSpan } from '../../../src/core/entities/span';

// Set up a proper tracer provider
const provider = new BasicTracerProvider();
trace.setGlobalTracerProvider(provider);

const tracer = trace.getTracer('mlflow-test-tracer', '1.0.0');

describe('Trace', () => {
  function createMockTraceInfo(): TraceInfo {
    return new TraceInfo({
      traceId: 'tr-12345',
      clientRequestId: 'client-request-id',
      traceLocation: createTraceLocationFromExperimentId('exp-123'),
      requestTime: Date.now(),
      state: TraceState.OK,
      executionDuration: 1000,
      requestPreview: '{"prompt": "Hello"}',
      responsePreview: '{"response": "Hi there"}',
      traceMetadata: { key: 'value' },
      tags: { env: 'test' },
      assessments: []
    });
  }

  function createMockTraceData(): TraceData {
    const span = tracer.startSpan('parent');
    const mlflowSpan = createMlflowSpan(span, 'tr-12345');
    span.end();
    return new TraceData([mlflowSpan]);
  }

  describe('constructor', () => {
    it('should create a Trace with info and data', () => {
      const traceInfo = createMockTraceInfo();
      const traceData = createMockTraceData();

      const trace = new Trace(traceInfo, traceData);

      expect(trace.info).toBe(traceInfo);
      expect(trace.data).toBe(traceData);
    });
  });

  describe('toJson/fromJson round-trip serialization', () => {
    it('should serialize and deserialize a complete trace correctly', () => {
      const originalTraceInfo = createMockTraceInfo();
      const originalTraceData = createMockTraceData();
      const originalTrace = new Trace(originalTraceInfo, originalTraceData);

      const json = originalTrace.toJson();

      // Verify JSON structure
      expect(json).toHaveProperty('info');
      expect(json).toHaveProperty('data');
      expect(json.info).toMatchObject({
        trace_id: 'tr-12345',
        request_time: expect.any(String),
        state: TraceState.OK
      });
      expect(json.data).toMatchObject({
        spans: [
          {
            name: 'parent',
            trace_id: expect.any(String),
            span_id: expect.any(String),
            parent_span_id: '',
            start_time_unix_nano: expect.any(BigInt),
            end_time_unix_nano: expect.any(BigInt),
            attributes: {
              'mlflow.spanType': 'UNKNOWN'
            },
            status: { code: 'STATUS_CODE_UNSET' },
            events: []
          }
        ]
      });

      // Round-trip test
      const recreatedTrace = Trace.fromJson(json);

      expect(recreatedTrace).toBeInstanceOf(Trace);
      expect(recreatedTrace.info).toBeInstanceOf(TraceInfo);
      expect(recreatedTrace.data).toBeInstanceOf(TraceData);

      // Verify that key properties are preserved
      expect(recreatedTrace.info.traceId).toBe(originalTrace.info.traceId);
      expect(recreatedTrace.info.state).toBe(originalTrace.info.state);
      expect(recreatedTrace.data.spans.length).toEqual(originalTrace.data.spans.length);

      // Verify round-trip JSON serialization matches
      expect(recreatedTrace.toJson()).toEqual(originalTrace.toJson());
    });
  });
});
