import { trace as otelTrace } from '@opentelemetry/api';
import { init } from '../../src/core/config';
import { flushTraces } from '../../src/core/provider';
import { getCurrentActiveSpan, startSpan, withSpan } from '../../src/core/api';
import { InMemoryTraceManager } from '../../src/core/trace_manager';
import { TEST_TRACKING_URI } from '../helper';

describe('init() with registerGlobalTracerProvider: false', () => {
  beforeAll(() => {
    init({
      trackingUri: TEST_TRACKING_URI,
      experimentId: '0',
      registerGlobalTracerProvider: false,
    });
  });

  afterAll(async () => {
    await flushTraces();
  });

  it('does not record spans that other libraries start through the global OpenTelemetry API', () => {
    const foreignSpan = otelTrace.getTracer('some-llm-sdk').startSpan('llm.request');
    foreignSpan.end();

    expect(foreignSpan.isRecording()).toBe(false);
    expect(
      InMemoryTraceManager.getInstance().getMlflowTraceIdFromOtelId(
        foreignSpan.spanContext().traceId,
      ),
    ).toBeNull();
  });

  it('still records spans created with the MLflow API', () => {
    const span = startSpan({ name: 'root' });
    const child = startSpan({ name: 'child', parent: span });
    child.end();
    span.end();

    expect(span.traceId).toMatch(/^tr-/);
    expect(child.traceId).toBe(span.traceId);
    expect(child.parentId).toBe(span.spanId);
  });

  it('still nests spans through the active context', async () => {
    let parentSpanId: string | undefined;
    let childParentId: string | null | undefined;

    await withSpan(
      async (parent) => {
        parentSpanId = parent.spanId;
        await Promise.resolve();
        expect(getCurrentActiveSpan()?.spanId).toBe(parent.spanId);
        withSpan(
          (child) => {
            childParentId = child.parentId;
          },
          { name: 'child' },
        );
      },
      { name: 'parent' },
    );

    expect(parentSpanId).toBeDefined();
    expect(childParentId).toBe(parentSpanId);
  });
});
