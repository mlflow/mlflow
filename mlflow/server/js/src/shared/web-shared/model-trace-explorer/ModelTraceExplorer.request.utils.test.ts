import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';

import { matchPredefinedError, PermissionError } from '../errors/PredefinedErrors';

jest.mock('./RoutingUtils', () => ({ getActiveWorkspace: () => 'synthetic' }));
jest.mock('../errors/PredefinedErrors', () => ({
  ...jest.requireActual<typeof import('../errors/PredefinedErrors')>('../errors/PredefinedErrors'),
  matchPredefinedError: jest.fn(),
}));

describe('trace request response parsers', () => {
  let fetchAPI: typeof import('./ModelTraceExplorer.request.utils').fetchAPI;
  let previousFetch: typeof fetch;
  const fetchMock = jest.fn<typeof fetch>();
  beforeEach(() => {
    previousFetch = globalThis.fetch;
    globalThis.fetch = fetchMock;
    fetchMock.mockReset();
    jest.mocked(matchPredefinedError).mockReset();
    jest.isolateModules(() => {
      // eslint-disable-next-line @typescript-eslint/no-require-imports -- Capture fetch after installing this test's mock.
      fetchAPI = require('./ModelTraceExplorer.request.utils').fetchAPI;
    });
    document.cookie = 'mlflow-request-header-X-Test=fixture';
  });
  afterEach(() => {
    globalThis.fetch = previousFetch;
    document.cookie = 'mlflow-request-header-X-Test=; Max-Age=0';
  });

  it('keeps native JSON, cookie/workspace headers, body serialization and AbortSignal by default', async () => {
    const json = jest.fn(async () => ({ okay: true }));
    fetchMock.mockResolvedValue({ ok: true, json } as unknown as Response);
    const controller = new AbortController();
    await expect(fetchAPI('/owned', 'POST', { prompt: 'café' }, controller.signal)).resolves.toEqual({ okay: true });
    expect(json).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith('/owned', {
      method: 'POST',
      signal: controller.signal,
      body: JSON.stringify({ prompt: 'café' }),
      headers: { 'Content-Type': 'application/json', 'X-Test': 'fixture', 'X-MLFLOW-WORKSPACE': 'synthetic' },
    });
  });

  it('hands an unread successful response to the requested parser', async () => {
    const response = { ok: true, json: jest.fn() } as unknown as Response;
    fetchMock.mockResolvedValue(response);
    const parse = jest.fn(async (_response: Response) => ({ parsed: true }));
    await expect(fetchAPI('/owned', 'GET', undefined, undefined, parse)).resolves.toEqual({ parsed: true });
    expect(parse).toHaveBeenCalledWith(response);
    expect(response.json).not.toHaveBeenCalled();
  });

  it('retains predefined error-message parsing before any custom success parser', async () => {
    const error = new PermissionError({ status: 403 });
    jest.mocked(matchPredefinedError).mockReturnValue(error);
    fetchMock.mockResolvedValue({ ok: false, json: async () => ({ message: 'denied' }) } as Response);
    const parse = jest.fn(async () => ({}));
    await expect(fetchAPI('/owned', 'GET', undefined, undefined, parse)).rejects.toBe(error);
    expect(error.message).toBe('denied');
    expect(parse).not.toHaveBeenCalled();
  });
});
