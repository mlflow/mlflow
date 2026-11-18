import { afterAll, beforeAll, beforeEach, describe, expect, it, jest } from '@jest/globals';

import { parseTraceResponseBody } from './traceResponseParsing';

jest.mock('./traceResponseParsing', () => ({
  ...jest.requireActual<typeof import('./traceResponseParsing')>('./traceResponseParsing'),
  parseTraceResponseBody: jest.fn(),
}));

describe('trace response worker protocol', () => {
  const originalHandler = globalThis.onmessage;
  const post = jest.spyOn(globalThis, 'postMessage').mockImplementation(() => {});
  beforeAll(async () => {
    await import('./traceResponse.worker');
  });
  beforeEach(() => {
    post.mockImplementation(() => {});
  });
  afterAll(() => {
    globalThis.onmessage = originalHandler;
    post.mockRestore();
  });

  it('announces readiness on load', () => {
    expect(post).toHaveBeenCalledWith({ type: 'ready' });
  });

  it('replies with the parsed value and originating request ID', () => {
    const value = { spans: [{ name: 'café 日本語 🙂' }] };
    const body = new ArrayBuffer(4);
    jest.mocked(parseTraceResponseBody).mockReturnValueOnce(value);
    globalThis.onmessage?.call(window, { data: { id: 23, body } } as MessageEvent);
    expect(parseTraceResponseBody).toHaveBeenLastCalledWith(body);
    expect(post).toHaveBeenLastCalledWith({ type: 'parsed', id: 23, value });
  });

  it.each(['artifact', 'full', 'batch'])('chunks large %s span lists without changing their contents', (shape) => {
    const spans = Array.from({ length: 96 }, (_, index) => ({
      name: `span-${index}`,
      attributes: { 'mlflow.spanInputs': JSON.stringify({ text: 'x'.repeat(32700) + ' café 日本語 🙂' }) },
    }));
    const value =
      shape === 'artifact'
        ? { spans, extra: { spans: ['user payload'] } }
        : shape === 'full'
          ? { trace: { trace_info: { trace_id: 'owned' }, spans } }
          : { traces: [{ spans }, { spans: [{ name: 'second trace' }] }] };
    const original = JSON.stringify(value);
    const messages: any[] = [];
    post.mockImplementation((message: unknown) => messages.push(JSON.parse(JSON.stringify(message))));
    jest.mocked(parseTraceResponseBody).mockReturnValueOnce(value);
    globalThis.onmessage?.call(window, { data: { id: 42, body: new ArrayBuffer(0) } } as MessageEvent);

    const chunks = messages.filter((message) => message.type === 'span-chunk');
    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks.every((message) => message.id === 42 && message.index === 0)).toBe(true);
    expect(Math.max(...chunks.map((message) => JSON.stringify(message).length))).toBeLessThan(2.1 * 1024 * 1024);
    const terminal = messages.at(-1);
    expect(terminal.type).toBe('parsed');
    const containers =
      shape === 'artifact' ? [terminal.value] : shape === 'full' ? [terminal.value.trace] : terminal.value.traces;
    expect(containers[0].spans).toEqual([]);
    containers[0].spans = chunks.flatMap((message) => message.spans);
    expect(JSON.stringify(terminal.value)).toBe(original);
    post.mockImplementation(() => {});
  });

  it('bounds span counts even when their attributes are empty', () => {
    const spans = Array.from({ length: 600 }, (_, index) => ({ name: `span-${index}` }));
    const start = post.mock.calls.length;
    jest.mocked(parseTraceResponseBody).mockReturnValueOnce({ spans });
    globalThis.onmessage?.call(window, { data: { id: 43, body: new ArrayBuffer(0) } } as MessageEvent);
    const chunks = post.mock.calls
      .slice(start)
      .map(([message]: any[]) => message)
      .filter((message) => message.type === 'span-chunk');
    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks.every((message) => message.spans.length <= 256)).toBe(true);
    expect(chunks.flatMap((message) => message.spans)).toEqual(spans);
  });

  it('counts OTLP string attributes and leaves a single oversized span intact', () => {
    const huge = { attributes: [{ key: 'mlflow.spanInputs', value: { string_value: '🙂'.repeat(1100000) } }] };
    const last = { attributes: [{ key: 'small', value: { string_value: 'café' } }] };
    const start = post.mock.calls.length;
    jest.mocked(parseTraceResponseBody).mockReturnValueOnce({ spans: [huge, last] });
    globalThis.onmessage?.call(window, { data: { id: 44, body: new ArrayBuffer(0) } } as MessageEvent);
    expect(post.mock.calls.slice(start).map(([message]) => message)).toEqual([
      { type: 'span-chunk', id: 44, index: 0, spans: [huge] },
      { type: 'span-chunk', id: 44, index: 0, spans: [last] },
      { type: 'parsed', id: 44, value: { spans: [] } },
    ]);
  });

  it.each([SyntaxError, RangeError, TypeError, Error])('sends the actual %p name and message', (ErrorClass) => {
    jest.mocked(parseTraceResponseBody).mockImplementation(() => {
      throw new ErrorClass('parse failed');
    });
    globalThis.onmessage?.call(window, { data: { id: 7, body: new ArrayBuffer(0) } } as MessageEvent);
    expect(post).toHaveBeenLastCalledWith({ type: 'failed', id: 7, name: ErrorClass.name, message: 'parse failed' });
  });
});
