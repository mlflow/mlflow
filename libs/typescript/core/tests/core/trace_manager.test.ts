import { InMemoryTraceManager } from '../../src/core/trace_manager';
import { Trace } from '../../src/core/entities/trace';
import { TraceInfo } from '../../src/core/entities/trace_info';
import { createTraceLocationFromExperimentId } from '../../src/core/entities/trace_location';
import { TraceState } from '../../src/core/entities/trace_state';
import { createTestSpan } from '../helper';
import { Span } from '../../src/core/entities/span';
import { TraceMetadataKey } from '../../src/core/constants';

/**
 * Helper function to create a test TraceInfo object
 */
function createTestTraceInfo(traceId: string): TraceInfo {
  return new TraceInfo({
    traceId,
    traceLocation: createTraceLocationFromExperimentId('1'),
    requestTime: Date.now(),
    state: TraceState.IN_PROGRESS,
    requestPreview: undefined,
    responsePreview: undefined,
    traceMetadata: {},
    tags: {}
  });
}

describe('InMemoryTraceManager', () => {
  beforeEach(() => {
    // Reset the singleton instance before each test
    InMemoryTraceManager.reset();
  });

  afterEach(() => {
    // Clean up after each test
    InMemoryTraceManager.reset();
  });

  it('should return the same instance when called multiple times', () => {
    const obj1 = InMemoryTraceManager.getInstance();
    const obj2 = InMemoryTraceManager.getInstance();
    expect(obj1).toBe(obj2);
  });

  it('should add spans and pop traces correctly', () => {
    const traceManager = InMemoryTraceManager.getInstance();

    // Add a new trace info
    const traceId = 'tr-1';
    const otelTraceId = '12345';
    traceManager.registerTrace(otelTraceId, createTestTraceInfo(traceId));

    // Add a span for a new trace
    const span11 = createTestSpan('test', traceId, 'span11');
    traceManager.registerSpan(span11);

    expect(traceManager.getTrace(traceId)).toBeTruthy();
    const trace1 = traceManager.getTrace(traceId);
    expect(trace1?.info.traceId).toBe(traceId);
    expect(trace1?.spanDict.size).toBe(1);

    // Add more spans to the same trace
    const span111 = createTestSpan('test', traceId, 'span111');
    const span112 = createTestSpan('test', traceId, 'span112');
    traceManager.registerSpan(span111);
    traceManager.registerSpan(span112);
    expect(trace1?.spanDict.size).toBe(3);

    // Pop the trace data
    const poppedTrace1 = traceManager.popTrace(otelTraceId);
    expect(poppedTrace1).toBeInstanceOf(Trace);
    expect(poppedTrace1?.info.traceId).toBe(traceId);
    expect(poppedTrace1?.data.spans.length).toBe(3);
    expect(traceManager.getTrace(traceId)).toBeNull();
    expect(poppedTrace1?.data.spans[0]).toBeInstanceOf(Span);
  });

  it('should truncate the request/response preview if it exceeds the max length', () => {
    const traceManager = InMemoryTraceManager.getInstance();
    const traceId = 'tr-1';
    const otelTraceId = '12345';
    traceManager.registerTrace(otelTraceId, createTestTraceInfo(traceId));

    const span = createTestSpan('test', traceId, 'span11');
    span.setInputs('a'.repeat(5000));
    traceManager.registerSpan(span);

    const trace = traceManager.popTrace(otelTraceId);
    expect(trace?.info.requestPreview).toHaveLength(1000);
    expect(trace?.info.traceMetadata[TraceMetadataKey.INPUTS]).toHaveLength(1000);
  });
});
