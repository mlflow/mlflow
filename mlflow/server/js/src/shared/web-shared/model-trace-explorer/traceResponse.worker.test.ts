import { afterAll, beforeAll, describe, expect, it, jest } from '@jest/globals';

import { parseTraceResponseBody } from './traceResponseParsing';

jest.mock('./traceResponseParsing', () => ({ parseTraceResponseBody: jest.fn() }));

describe('trace response worker protocol', () => {
  const originalHandler = globalThis.onmessage;
  const post = jest.spyOn(globalThis, 'postMessage').mockImplementation(() => {});
  beforeAll(async () => {
    await import('./traceResponse.worker');
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

  it.each([SyntaxError, RangeError, TypeError, Error])('sends the actual %p name and message', (ErrorClass) => {
    jest.mocked(parseTraceResponseBody).mockImplementation(() => {
      throw new ErrorClass('parse failed');
    });
    globalThis.onmessage?.call(window, { data: { id: 7, body: new ArrayBuffer(0) } } as MessageEvent);
    expect(post).toHaveBeenLastCalledWith({ type: 'failed', id: 7, name: ErrorClass.name, message: 'parse failed' });
  });
});
