import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';

import { getBatchTracesV4, getExperimentTraceV3, getTraceV3, getTraceV3Data, getTraceV3Info } from './api';
import { fetchAPI } from './ModelTraceExplorer.request.utils';
import { createTraceResponseParser, TraceResponseWorkerLostError } from './traceResponseParser';

jest.mock('./ModelTraceExplorer.request.utils', () => ({ fetchAPI: jest.fn(), getAjaxUrl: (url: string) => url }));
jest.mock('./traceResponseParser', () => ({
  createTraceResponseParser: jest.fn(),
  TraceResponseWorkerLostError: class extends Error {},
}));
const fetchMock = jest.mocked(fetchAPI);
const factory = jest.mocked(createTraceResponseParser);

describe('trace body response parsing', () => {
  const response = {} as Response;
  const parse = jest.fn<(response: Response, signal?: AbortSignal) => Promise<unknown>>();
  const dispose = jest.fn();
  beforeEach(() => {
    factory.mockReset().mockReturnValue({ parse, dispose });
    parse.mockReset().mockResolvedValue({ spans: [] });
    dispose.mockClear();
    fetchMock
      .mockReset()
      .mockImplementation(async (_url, _method, _body, _signal, parseResponse) =>
        parseResponse ? parseResponse(response) : { native: true },
      );
  });
  afterEach(() => {
    window.dispatchEvent(new Event('pagehide'));
  });

  it('uses one parser for full, artifact and batch bodies while leaving trace info on native JSON', async () => {
    await expect(getExperimentTraceV3({ traceId: 'owned' })).resolves.toEqual({ spans: [] });
    await expect(getTraceV3Data('owned')).resolves.toEqual({ spans: [] });
    await expect(getBatchTracesV4({ traceIds: ['owned'], traceLocation: 'synthetic' })).resolves.toEqual({ spans: [] });
    expect(factory).toHaveBeenCalledTimes(1);
    expect(parse).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls.slice(0, 3).every((call) => call[1] === 'GET' && typeof call[4] === 'function')).toBe(
      true,
    );
    await expect(getTraceV3Info('owned')).resolves.toEqual({ native: true });
    expect(fetchMock.mock.calls[3][4]).toBeUndefined();
  });

  it('re-fetches a lost transferred GET body once using the original native parser', async () => {
    parse.mockRejectedValue(new TraceResponseWorkerLostError());
    await expect(getTraceV3Data('owned')).resolves.toEqual({ native: true });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0][0]).toBe(fetchMock.mock.calls[1][0]);
    expect(fetchMock.mock.calls[1].slice(1)).toEqual(['GET', undefined, undefined]);
    expect(parse).toHaveBeenCalledTimes(1);
  });

  it('forwards caller cancellation through each trace body transport and parser', async () => {
    const { signal } = new AbortController();
    await getExperimentTraceV3({ traceId: 'owned', signal });
    await getTraceV3Data('owned', signal);
    await getBatchTracesV4({ traceIds: ['owned'], traceLocation: 'synthetic', signal });
    await getTraceV3('owned', signal);
    expect(fetchMock.mock.calls).toHaveLength(5);
    expect(fetchMock.mock.calls.every((call) => call[3] === signal)).toBe(true);
    expect(parse).toHaveBeenCalledTimes(4);
    expect(parse.mock.calls.every((call) => call[1] === signal)).toBe(true);
  });

  it('does not retry a lost body when its caller has canceled', async () => {
    const controller = new AbortController();
    const lost = new TraceResponseWorkerLostError();
    parse.mockImplementation(async () => {
      controller.abort();
      throw lost;
    });
    await expect(getTraceV3Data('owned', controller.signal)).rejects.toBe(lost);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it.each([new SyntaxError('invalid JSON'), new DOMException('Aborted', 'AbortError'), new Error('network')])(
    'does not retry parsing, abort or network failures: %p',
    async (error) => {
      parse.mockRejectedValue(error);
      await expect(getTraceV3Data('owned')).rejects.toBe(error);
      expect(fetchMock).toHaveBeenCalledTimes(1);
    },
  );

  it('disposes the page parser and recreates it after a pagehide restoration', async () => {
    await getTraceV3Data('owned');
    window.dispatchEvent(new Event('pagehide'));
    expect(dispose).toHaveBeenCalledTimes(1);
    await getTraceV3Data('owned');
    expect(factory).toHaveBeenCalledTimes(2);
  });
});
