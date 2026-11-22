import { ROOT_CONTEXT, TraceFlags, trace as otelTrace } from '@opentelemetry/api';
import { ExportResult, ExportResultCode } from '@opentelemetry/core';
import {
  BasicTracerProvider,
  ReadableSpan as OTelReadableSpan,
  SpanExporter,
} from '@opentelemetry/sdk-trace-base';

jest.mock('../../src/core/config', () => ({
  getConfig: () => ({ experimentId: '0', trackingUri: 'http://localhost:5000' }),
}));

jest.mock('../../src/core/utils/environment', () => ({
  resolveEnvironmentMetadata: () => ({}),
}));

import { SpanAttributeKey } from '../../src/core/constants';
import { Trace } from '../../src/core/entities/trace';
import { InMemoryTraceManager } from '../../src/core/trace_manager';
import { MlflowSpanProcessor } from '../../src/exporters/mlflow';

const REMOTE_TRACE_ID = '0af7651916cd43dd8448eb211c80319c';
const REMOTE_SPAN_ID = 'b7ad6b7169203331';

class RecordingExporter implements SpanExporter {
  traces: Trace[] = [];

  export(spans: OTelReadableSpan[], resultCallback: (result: ExportResult) => void): void {
    for (const span of spans) {
      const trace = InMemoryTraceManager.getInstance().popTrace(span.spanContext().traceId);
      if (trace) {
        this.traces.push(trace);
      }
    }
    resultCallback({ code: ExportResultCode.SUCCESS });
  }

  shutdown(): Promise<void> {
    return Promise.resolve();
  }

  forceFlush(): Promise<void> {
    return Promise.resolve();
  }
}

describe('MlflowSpanProcessor', () => {
  let exporter: RecordingExporter;
  let tracer: ReturnType<BasicTracerProvider['getTracer']>;

  beforeEach(() => {
    InMemoryTraceManager.reset();
    exporter = new RecordingExporter();
    const provider = new BasicTracerProvider({
      spanProcessors: [new MlflowSpanProcessor(exporter)],
    });
    tracer = provider.getTracer('mlflow-test');
  });

  it('exports a trace whose root span has a remote parent', () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const remoteCtx = otelTrace.setSpanContext(ROOT_CONTEXT, {
      traceId: REMOTE_TRACE_ID,
      spanId: REMOTE_SPAN_ID,
      traceFlags: TraceFlags.SAMPLED,
      isRemote: true,
    });

    const root = tracer.startSpan('remote-parent-root', {}, remoteCtx);
    root.setAttribute(SpanAttributeKey.INPUTS, JSON.stringify({ question: 'hi' }));
    root.setAttribute(SpanAttributeKey.OUTPUTS, JSON.stringify('hello'));
    const child = tracer.startSpan('child', {}, otelTrace.setSpan(remoteCtx, root));
    child.end();
    root.end();

    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();

    expect(exporter.traces).toHaveLength(1);
    const exported = exporter.traces[0];
    expect(exported.info.traceId).toBe(`tr-${REMOTE_TRACE_ID}`);
    expect(exported.info.requestPreview).toBe('{"question":"hi"}');
    expect(exported.info.responsePreview).toBe('"hello"');

    const spans = Object.fromEntries(exported.data.spans.map((s) => [s.name, s]));
    expect(Object.keys(spans).sort()).toEqual(['child', 'remote-parent-root']);
    expect(spans['remote-parent-root'].parentId).toBeNull();
    expect(spans['child'].parentId).toBe(spans['remote-parent-root'].spanId);
    // The OTel span keeps its remote parent so it still links to the caller's distributed trace
    expect(spans['remote-parent-root']._span.parentSpanContext?.spanId).toBe(REMOTE_SPAN_ID);
  });

  it('still exports a trace whose root span has no parent', () => {
    tracer.startSpan('local-root', {}, ROOT_CONTEXT).end();

    expect(exporter.traces.map((t) => t.data.spans.map((s) => s.name))).toEqual([['local-root']]);
  });
});
