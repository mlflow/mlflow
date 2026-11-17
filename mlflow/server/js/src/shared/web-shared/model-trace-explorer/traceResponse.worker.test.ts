import { afterAll, beforeAll, describe, expect, it, jest } from '@jest/globals';

import { parseTraceResponseBody } from './traceResponseParsing';

jest.mock('./traceResponseParsing', () => ({ parseTraceResponseBody: jest.fn() }));

describe('trace response worker failures', () => {
  const originalHandler = globalThis.onmessage;
  const post = jest.spyOn(globalThis, 'postMessage').mockImplementation(() => {});
  beforeAll(async () => {
    await import('./traceResponse.worker');
  });
  afterAll(() => {
    globalThis.onmessage = originalHandler;
    post.mockRestore();
  });

  it.each([SyntaxError, RangeError, TypeError, Error])('sends the actual %p name and message', (ErrorClass) => {
    jest.mocked(parseTraceResponseBody).mockImplementation(() => {
      throw new ErrorClass('parse failed');
    });
    globalThis.onmessage?.call(window, { data: { id: 7, body: new ArrayBuffer(0) } } as MessageEvent);
    expect(post).toHaveBeenLastCalledWith({ type: 'failed', id: 7, name: ErrorClass.name, message: 'parse failed' });
  });
});
