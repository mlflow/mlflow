import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { TextDecoder } from 'util';

import { createTraceResponseWorker } from './createTraceResponseWorker';
import { createTraceResponseParser, TraceResponseWorkerLostError } from './traceResponseParser';
import { parseTraceResponseBody, type TraceResponseMessage } from './traceResponseParsing';

jest.mock('./createTraceResponseWorker', () => ({ createTraceResponseWorker: jest.fn() }));
const factory = jest.mocked(createTraceResponseWorker);
const encode = (text: string) => Uint8Array.from(Buffer.from(text)).buffer;
const response = (value: unknown) => ({
  json: jest.fn(async () => value),
  arrayBuffer: jest.fn(async () => encode(JSON.stringify(value))),
});
const body = (value: unknown) => response(value) as unknown as Response;
const large = { content: 'Привет café 日本語 🙂 '.repeat(5000) };

class FakeWorker {
  onmessage?: (event: MessageEvent<TraceResponseMessage>) => void;
  onerror?: () => void;
  onmessageerror?: () => void;
  postMessage = jest.fn<(request: { id: number; body: ArrayBuffer }, transfer: Transferable[]) => void>();
  terminate = jest.fn();
  send(data: TraceResponseMessage) {
    this.onmessage?.({ data } as MessageEvent<TraceResponseMessage>);
  }
}

describe('trace response parsing', () => {
  const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
  const originalDecoder = Object.getOwnPropertyDescriptor(globalThis, 'TextDecoder');
  const originalWorker = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  let worker: FakeWorker;
  let parser: ReturnType<typeof createTraceResponseParser>;
  beforeAll(() => {
    Object.defineProperty(globalThis, 'TextDecoder', { configurable: true, value: TextDecoder });
  });
  afterAll(() => {
    warn.mockRestore();
    if (originalWorker) Object.defineProperty(globalThis, 'Worker', originalWorker);
    else delete (globalThis as { Worker?: unknown }).Worker;
    if (originalDecoder) Object.defineProperty(globalThis, 'TextDecoder', originalDecoder);
    else delete (globalThis as { TextDecoder?: unknown }).TextDecoder;
  });
  beforeEach(async () => {
    warn.mockClear();
    Object.defineProperty(globalThis, 'Worker', { configurable: true, value: FakeWorker });
    jest.useFakeTimers();
    worker = new FakeWorker();
    factory.mockReset().mockReturnValue(worker as unknown as Worker);
    parser = createTraceResponseParser();
    await import('./createTraceResponseWorker');
  });
  afterEach(() => {
    parser.dispose();
    jest.useRealTimers();
  });

  it('retains UTF-8, BOM and JSON scalar semantics and rejects malformed JSON', () => {
    for (const value of [large, null, true, 42, ['café', '🙂']]) {
      expect(parseTraceResponseBody(encode('\uFEFF' + JSON.stringify(value)))).toEqual(value);
    }
    expect(() => parseTraceResponseBody(encode('{invalid'))).toThrow(SyntaxError);
  });

  it('uses native JSON while the worker is starting, absent or failed to load', async () => {
    const input = response(large);
    await expect(parser.parse(input as unknown as Response)).resolves.toEqual(large);
    expect(input.json).toHaveBeenCalledTimes(1);
    expect(input.arrayBuffer).not.toHaveBeenCalled();
    parser.dispose();
    factory.mockImplementation(() => {
      throw Error('CSP');
    });
    parser = createTraceResponseParser();
    await expect(parser.parse(body(large))).resolves.toEqual(large);
    expect(jest.getTimerCount()).toBe(0);
    expect(warn).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({ cause: new Error('CSP') }));
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it('does not load worker code when Workers are unavailable', async () => {
    parser.dispose();
    factory.mockClear();
    delete (globalThis as { Worker?: unknown }).Worker;
    parser = createTraceResponseParser();
    await expect(parser.parse(body(large))).resolves.toEqual(large);
    expect(factory).not.toHaveBeenCalled();
    expect(jest.getTimerCount()).toBe(0);
  });

  it('does not create a worker if disposed before the factory chunk arrives', async () => {
    parser.dispose();
    factory.mockClear();
    parser = createTraceResponseParser();
    parser.dispose();
    await import('./createTraceResponseWorker');
    expect(factory).not.toHaveBeenCalled();
    expect(jest.getTimerCount()).toBe(0);
  });

  it('keeps small responses on this thread and posts large buffers with their transfer list', async () => {
    worker.send({ type: 'ready' });
    await expect(parser.parse(body({ content: 'small' }))).resolves.toEqual({ content: 'small' });
    expect(worker.postMessage).not.toHaveBeenCalled();
    const result = parser.parse(body(large));
    await Promise.resolve();
    const [request, transfer] = worker.postMessage.mock.calls[0];
    expect(transfer).toEqual([request.body]);
    worker.send({ type: 'parsed', id: request.id, value: parseTraceResponseBody(request.body) });
    await expect(result).resolves.toEqual(large);
    expect(jest.getTimerCount()).toBe(0);
  });

  it.each([65_535, 65_536])('uses the worker only at the exact64KiB boundary: %i bytes', async (bytes) => {
    worker.send({ type: 'ready' });
    const value = 'a'.repeat(bytes - 2);
    const result = parser.parse(body(value));
    await Promise.resolve();
    if (bytes < 65_536) {
      expect(worker.postMessage).not.toHaveBeenCalled();
    } else {
      const [{ id, body: transferred }] = worker.postMessage.mock.calls[0];
      expect(transferred.byteLength).toBe(bytes);
      worker.send({ type: 'parsed', id, value });
    }
    await expect(result).resolves.toBe(value);
  });

  it('suppresses a native fallback that finishes after disposal', async () => {
    let complete!: (value: unknown) => void;
    const result = parser.parse({
      json: () =>
        new Promise((resolve) => {
          complete = resolve;
        }),
    } as Response);
    parser.dispose();
    complete(large);
    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('rejects every transferred response when their shared worker fails', async () => {
    worker.send({ type: 'ready' });
    const first = parser.parse(body(large));
    const second = parser.parse(body(large));
    await Promise.resolve();
    worker.onerror?.();
    await expect(first).rejects.toBeInstanceOf(TraceResponseWorkerLostError);
    await expect(second).rejects.toBeInstanceOf(TraceResponseWorkerLostError);
    expect(jest.getTimerCount()).toBe(0);
  });

  it('correlates overlapping replies and preserves a JSON syntax error without retrying it', async () => {
    worker.send({ type: 'ready' });
    const first = parser.parse(body(large));
    const second = parser.parse(body(large));
    await Promise.resolve();
    const [a, b] = worker.postMessage.mock.calls.map(([request]) => request);
    worker.send({ type: 'failed', id: b.id, name: 'SyntaxError', message: 'bad JSON' });
    await expect(second).rejects.toThrow(SyntaxError);
    worker.send({ type: 'parsed', id: a.id, value: large });
    await expect(first).resolves.toEqual(large);
    expect(worker.terminate).not.toHaveBeenCalled();
  });

  it.each([RangeError, TypeError, Error])(
    'preserves a worker %p without labeling it a syntax error',
    async (ErrorClass) => {
      worker.send({ type: 'ready' });
      const result = parser.parse(body(large));
      await Promise.resolve();
      const [{ id }] = worker.postMessage.mock.calls[0];
      worker.send({ type: 'failed', id, name: ErrorClass.name, message: 'parse failed' });
      await expect(result).rejects.toBeInstanceOf(ErrorClass);
      await expect(result).rejects.toMatchObject({ name: ErrorClass.name, message: 'parse failed' });
      expect(jest.getTimerCount()).toBe(0);
    },
  );

  it.each(['error', 'messageerror', 'post failure'])(
    'rejects a lost body on %s and falls back for future responses',
    async (failure) => {
      worker.send({ type: 'ready' });
      if (failure === 'post failure') {
        worker.postMessage.mockImplementation(() => {
          throw Error('post failed');
        });
      }
      const result = parser.parse(body(large));
      await Promise.resolve();
      if (failure === 'error') worker.onerror?.();
      if (failure === 'messageerror') worker.onmessageerror?.();
      await expect(result).rejects.toBeInstanceOf(TraceResponseWorkerLostError);
      if (failure === 'post failure') {
        await expect(result).rejects.toMatchObject({ cause: new Error('post failed') });
      }
      expect(worker.terminate).toHaveBeenCalledTimes(1);
      expect(warn).toHaveBeenCalledTimes(1);
      expect(warn).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({ message: expect.any(String) }));
      await expect(parser.parse(body(large))).resolves.toEqual(large);
      expect(jest.getTimerCount()).toBe(0);
    },
  );

  it('retires a timed-out worker after unrelated requests finish and parses future responses natively', async () => {
    worker.send({ type: 'ready' });
    const first = parser.parse(body(large));
    await Promise.resolve();
    jest.advanceTimersByTime(5_000);
    const second = parser.parse(body(large));
    await Promise.resolve();
    const [a, b] = worker.postMessage.mock.calls.map(([request]) => request);
    jest.advanceTimersByTime(5_000);
    await expect(first).rejects.toBeInstanceOf(TraceResponseWorkerLostError);
    expect(worker.terminate).not.toHaveBeenCalled();
    worker.send({ type: 'parsed', id: b.id, value: large });
    await expect(second).resolves.toEqual(large);
    worker.send({ type: 'parsed', id: a.id, value: large });
    expect(worker.terminate).toHaveBeenCalledTimes(1);
    const third = response(large);
    await expect(parser.parse(third as unknown as Response)).resolves.toEqual(large);
    expect(third.json).toHaveBeenCalledTimes(1);
    expect(worker.postMessage).toHaveBeenCalledTimes(2);
    expect(jest.getTimerCount()).toBe(0);
  });

  it('gives up on a silent startup without delaying responses', async () => {
    jest.advanceTimersByTime(10_000);
    expect(worker.terminate).toHaveBeenCalledTimes(1);
    worker.send({ type: 'ready' });
    await expect(parser.parse(body(large))).resolves.toEqual(large);
    expect(worker.postMessage).not.toHaveBeenCalled();
  });

  it('aborts pending parsing, removes its listener/timer, and ignores a late reply', async () => {
    worker.send({ type: 'ready' });
    const controller = new AbortController();
    const remove = jest.spyOn(controller.signal, 'removeEventListener');
    const result = parser.parse(body(large), controller.signal);
    await Promise.resolve();
    const [{ id }] = worker.postMessage.mock.calls[0];
    controller.abort();
    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
    worker.send({ type: 'parsed', id, value: large });
    expect(remove).toHaveBeenCalledWith('abort', expect.any(Function));
    expect(jest.getTimerCount()).toBe(0);
    await expect(parser.parse(body(large), controller.signal)).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('disposes pending parses and prevents later reuse', async () => {
    worker.send({ type: 'ready' });
    const result = parser.parse(body(large));
    await Promise.resolve();
    parser.dispose();
    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
    await expect(parser.parse(body(large))).rejects.toMatchObject({ name: 'AbortError' });
    expect(jest.getTimerCount()).toBe(0);
    expect(warn).not.toHaveBeenCalled();
  });
});
