import type { SpanExporter, ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { MLflowSpanProcessor } from '../src/processor';

function makeSpan(attributes: Record<string, unknown> = {}): ReadableSpan {
  return {
    attributes,
    name: 'test-span',
    spanContext: () => ({
      traceId: 'abc123',
      spanId: Math.random().toString(16).slice(2, 18),
      traceFlags: 1,
    }),
    duration: [0, 0],
    startTime: [0, 0],
    endTime: [0, 0],
    ended: true,
    status: { code: 0 },
    kind: 0,
    resource: { attributes: {} },
    instrumentationLibrary: { name: 'test' },
    events: [],
    links: [],
    parentSpanId: undefined,
  } as unknown as ReadableSpan;
}

function getAttr(span: ReadableSpan, key: string): unknown {
  return (span.attributes as Record<string, unknown>)[key];
}

function createMockExporter(): SpanExporter & {
  exportMock: jest.Mock;
  shutdownMock: jest.Mock;
} {
  const exportMock = jest.fn((_spans: ReadableSpan[], cb: (result: { code: number }) => void) => {
    cb({ code: 0 });
  });
  const shutdownMock = jest.fn(() => Promise.resolve());

  return {
    export: exportMock,
    shutdown: shutdownMock,
    exportMock,
    shutdownMock,
  };
}

describe('MLflowSpanProcessor', () => {
  it('translates span attributes in onEnd before export', async () => {
    const mock = createMockExporter();
    const processor = new MLflowSpanProcessor(mock);

    const span = makeSpan({
      'ai.operationId': 'ai.generateText',
      'ai.model.id': 'gpt-4',
    });

    processor.onEnd(span);

    // Attributes should be translated immediately (before batched export)
    expect(getAttr(span, 'mlflow.spanType')).toBe('LLM');
    expect(getAttr(span, 'mlflow.llm.model')).toBe('gpt-4');

    await processor.shutdown();
  });

  it('exports translated spans via the inner exporter', async () => {
    const mock = createMockExporter();
    const processor = new MLflowSpanProcessor(mock);

    const span = makeSpan({
      'ai.operationId': 'ai.generateText',
      'ai.model.id': 'gpt-4',
    });

    processor.onEnd(span);
    await processor.forceFlush();

    // BatchSpanProcessor should have flushed the span to the exporter
    expect(mock.exportMock).toHaveBeenCalled();
    const exportedSpans = mock.exportMock.mock.calls[0][0] as ReadableSpan[];
    expect(getAttr(exportedSpans[0], 'mlflow.spanType')).toBe('LLM');

    await processor.shutdown();
  });

  it('passes non-AI spans through unchanged', async () => {
    const mock = createMockExporter();
    const processor = new MLflowSpanProcessor(mock);

    const span = makeSpan({
      'http.method': 'GET',
      'http.url': 'https://example.com',
    });
    const originalAttrs = { ...span.attributes };

    processor.onEnd(span);

    expect(span.attributes).toEqual(originalAttrs);

    await processor.shutdown();
  });

  it('shutdown resolves cleanly', async () => {
    const mock = createMockExporter();
    const processor = new MLflowSpanProcessor(mock);

    await expect(processor.shutdown()).resolves.toBeUndefined();
  });

  it('forceFlush resolves cleanly', async () => {
    const mock = createMockExporter();
    const processor = new MLflowSpanProcessor(mock);

    await expect(processor.forceFlush()).resolves.toBeUndefined();

    await processor.shutdown();
  });

  it('does not drop spans when translation fails for some', async () => {
    const mock = createMockExporter();
    const processor = new MLflowSpanProcessor(mock);

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

    // Both spans should be processed without throwing
    processor.onEnd(badSpan);
    processor.onEnd(goodSpan);

    // Good span should still be translated
    expect(getAttr(goodSpan, 'mlflow.spanType')).toBe('LLM');

    await processor.shutdown();
  });
});
