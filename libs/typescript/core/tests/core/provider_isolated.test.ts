import { context, trace as otelTrace } from '@opentelemetry/api';
import { BasicTracerProvider } from '@opentelemetry/sdk-trace-base';
import { MlflowClient } from '../../src/clients';
import { init } from '../../src/core/config';
import { flushTraces } from '../../src/core/provider';
import { getCurrentActiveSpan, startSpan, updateCurrentTrace, withSpan } from '../../src/core/api';
import { InMemoryTraceManager } from '../../src/core/trace_manager';
import { TEST_TRACKING_URI } from '../helper';

beforeEach(() => {
  jest.spyOn(MlflowClient.prototype, 'createTrace').mockImplementation(async (trace) => trace);
  jest.spyOn(MlflowClient.prototype, 'uploadTraceData').mockResolvedValue(undefined);
  jest.spyOn(MlflowClient.prototype, 'exportOtlpSpans').mockResolvedValue(undefined);
});

afterEach(async () => {
  await flushTraces();
  jest.restoreAllMocks();
});

it('records MLflow spans without replacing an existing global provider', () => {
  const host = new BasicTracerProvider();
  expect(otelTrace.setGlobalTracerProvider(host)).toBe(true);
  const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
  try {
    init({ trackingUri: TEST_TRACKING_URI, experimentId: '0' });
    const span = startSpan({ name: 'mlflow root' });
    span.end();
    expect(span.traceId).toMatch(/^tr-/);
    expect(otelTrace.getTracer('host')).toBe(host.getTracer('host'));
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('Global tracer registration failed'));
  } finally {
    otelTrace.disable();
  }
});

describe('init() with registerGlobalTracerProvider: false', () => {
  beforeAll(() => {
    init({
      trackingUri: TEST_TRACKING_URI,
      experimentId: '0',
      registerGlobalTracerProvider: false,
    });
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

  it('starts new traces under a foreign active span', () => {
    const foreignSpan = new BasicTracerProvider().getTracer('host').startSpan('host');
    try {
      context.with(otelTrace.setSpan(context.active(), foreignSpan), () => {
        const span = startSpan({ name: 'root' });
        span.end();
        expect(span.traceId).toMatch(/^tr-/);
        expect(span.parentId).toBeNull();
        withSpan(
          (active) => {
            expect(active.traceId).toMatch(/^tr-/);
            expect(active.parentId).toBeNull();
            updateCurrentTrace({ tags: { isolated: 'yes' } });
            expect(
              InMemoryTraceManager.getInstance().getTrace(active.traceId)?.info.tags.isolated,
            ).toBe('yes');
          },
          { name: 'active root' },
        );
      });
    } finally {
      foreignSpan.end();
    }
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
