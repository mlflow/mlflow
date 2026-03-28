import type { SpanExporter, ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { MLflowSpanExporter } from '../src/exporter';

function makeSpan(attributes: Record<string, unknown> = {}): ReadableSpan {
  return {
    attributes,
    name: 'test-span',
    spanContext: () => ({
      traceId: 'abc123',
      spanId: Math.random().toString(16).slice(2, 18),
      traceFlags: 1,
    }),
  } as unknown as ReadableSpan;
}

function getAttr(span: ReadableSpan, key: string): unknown {
  return (span.attributes as Record<string, unknown>)[key];
}

function createMockExporter(): SpanExporter & {
  exportMock: jest.Mock;
  shutdownMock: jest.Mock;
  forceFlushMock: jest.Mock;
} {
  const exportMock = jest.fn(
    (_spans: ReadableSpan[], cb: (result: { code: number }) => void) => {
      cb({ code: 0 });
    }
  );
  const shutdownMock = jest.fn(() => Promise.resolve());
  const forceFlushMock = jest.fn(() => Promise.resolve());

  return {
    export: exportMock,
    shutdown: shutdownMock,
    forceFlush: forceFlushMock,
    exportMock,
    shutdownMock,
    forceFlushMock,
  };
}

describe('MLflowSpanExporter', () => {
  it('translates spans before delegating to inner exporter', () => {
    const mock = createMockExporter();
    const exporter = new MLflowSpanExporter(mock);

    const span = makeSpan({
      'ai.operationId': 'ai.generateText',
      'ai.model.id': 'gpt-4',
    });

    const callback = jest.fn();
    exporter.export([span], callback);

    // Inner exporter was called
    expect(mock.exportMock).toHaveBeenCalledTimes(1);

    // Spans passed to inner exporter have mlflow.* attributes
    const exportedSpans = mock.exportMock.mock.calls[0][0] as ReadableSpan[];
    expect(getAttr(exportedSpans[0], 'mlflow.spanType')).toBe('LLM');
    expect(getAttr(exportedSpans[0], 'mlflow.llm.model')).toBe('gpt-4');

    // Callback was forwarded
    expect(callback).toHaveBeenCalledWith({ code: 0 });
  });

  it('forwards error results from inner exporter', () => {
    const mock = createMockExporter();
    const error = new Error('export failed');
    mock.exportMock.mockImplementation(
      (_spans: ReadableSpan[], cb: (result: { code: number; error?: Error }) => void) => {
        cb({ code: 1, error });
      }
    );

    const exporter = new MLflowSpanExporter(mock);
    const callback = jest.fn();
    exporter.export([makeSpan()], callback);

    expect(callback).toHaveBeenCalledWith({ code: 1, error });
  });

  it('delegates shutdown to inner exporter', async () => {
    const mock = createMockExporter();
    const exporter = new MLflowSpanExporter(mock);

    await exporter.shutdown();

    expect(mock.shutdownMock).toHaveBeenCalledTimes(1);
  });

  it('delegates forceFlush to inner exporter', async () => {
    const mock = createMockExporter();
    const exporter = new MLflowSpanExporter(mock);

    await exporter.forceFlush();

    expect(mock.forceFlushMock).toHaveBeenCalledTimes(1);
  });

  it('resolves forceFlush when inner exporter lacks it', async () => {
    const mock = createMockExporter();
    // Remove forceFlush to simulate an exporter that doesn't implement it
    delete (mock as unknown as Record<string, unknown>)['forceFlush'];

    const exporter = new MLflowSpanExporter(mock);

    // Should resolve without error
    await expect(exporter.forceFlush()).resolves.toBeUndefined();
  });

  it('passes non-AI spans through unchanged', () => {
    const mock = createMockExporter();
    const exporter = new MLflowSpanExporter(mock);

    const span = makeSpan({
      'http.method': 'GET',
      'http.url': 'https://example.com',
    });
    const originalAttrs = { ...span.attributes };

    exporter.export([span], jest.fn());

    const exportedSpans = mock.exportMock.mock.calls[0][0] as ReadableSpan[];
    expect(exportedSpans[0].attributes).toEqual(originalAttrs);
  });

  it('delegates empty span array to inner exporter', () => {
    const mock = createMockExporter();
    const exporter = new MLflowSpanExporter(mock);
    const callback = jest.fn();

    exporter.export([], callback);

    expect(mock.exportMock).toHaveBeenCalledTimes(1);
    expect(mock.exportMock.mock.calls[0][0]).toHaveLength(0);
    expect(callback).toHaveBeenCalledWith({ code: 0 });
  });

  it('exports all spans even when translation fails for some', () => {
    const mock = createMockExporter();
    const exporter = new MLflowSpanExporter(mock);

    const badSpan = makeSpan({});
    Object.defineProperty(badSpan, 'attributes', {
      get() {
        throw new Error('boom');
      },
    });

    const goodSpan = makeSpan({
      'ai.operationId': 'ai.generateText',
      'ai.model.id': 'gpt-4',
    });

    exporter.export([badSpan, goodSpan], jest.fn());

    // Inner exporter still called with both spans
    expect(mock.exportMock).toHaveBeenCalledTimes(1);
    const exportedSpans = mock.exportMock.mock.calls[0][0] as ReadableSpan[];
    expect(exportedSpans).toHaveLength(2);
  });
});
