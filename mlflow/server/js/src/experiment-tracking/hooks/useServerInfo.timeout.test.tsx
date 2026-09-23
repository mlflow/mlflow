import { describe, beforeEach, afterEach, test, expect, jest } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';

import { SERVER_INFO_TIMEOUT_MS, useServerInfo } from './useServerInfo';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

const SERVER_RESPONSE = {
  store_type: 'SqlStore',
  workspaces_enabled: true,
  trace_archival_enabled: false,
  multipart_uploads_enabled: false,
  multipart_downloads_enabled: false,
};

const jsonResponse = (body: unknown) => ({ ok: true, status: 200, json: async () => body }) as unknown as Response;

describe('useServerInfo when the server-info request hangs', () => {
  // Regression guard: MlflowRouter holds the whole UI on a skeleton until this query settles, so a
  // request that never responds leaves the app unusable with no error and no route. fetchServerInfo
  // aborts after SERVER_INFO_TIMEOUT_MS so a hang takes the same path a 500 already takes.
  //
  // These stub `global.fetch` rather than `fetchAPI`, deliberately. The fix depends on `fetchAPI`
  // forwarding `signal` into the `RequestInit` it builds; mocking `fetchAPI` would hide exactly
  // that, and the tests would still pass if the passthrough were removed.
  let queryClient: QueryClient;
  let originalFetch: typeof global.fetch;
  let fetchMock: jest.Mock;

  const hangUntilAborted = (_url: unknown, init?: RequestInit) =>
    new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    });

  beforeEach(() => {
    jest.useFakeTimers();
    originalFetch = global.fetch;
    fetchMock = jest.fn(hangUntilAborted as never);
    global.fetch = fetchMock as unknown as typeof global.fetch;
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  });

  afterEach(() => {
    global.fetch = originalFetch;
    jest.useRealTimers();
    queryClient.clear();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  test('sends an abort signal with the request', async () => {
    renderHook(() => useServerInfo(), { wrapper });

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalled();
    });

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.signal).toBeInstanceOf(AbortSignal);
    expect(init.signal?.aborted).toBe(false);
  });

  test('falls back to the default response once the timeout fires', async () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const { result } = renderHook(() => useServerInfo(), { wrapper });

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalled();
    });
    expect(result.current.isLoading).toBe(true);

    jest.advanceTimersByTime(SERVER_INFO_TIMEOUT_MS);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });
    expect(result.current.data?.store_type).toBe('');
    expect(result.current.data?.workspaces_enabled).toBe(false);
    // The fallback is otherwise indistinguishable from a server that reports these defaults.
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('did not respond'));

    warn.mockRestore();
  });

  test('keeps a response that arrives just before the deadline, and clears the abort timer', async () => {
    // Counting pending timers would be ambiguous here, since React Query schedules its own.
    // Pin the specific timer this code arms instead.
    const setTimeoutSpy = jest.spyOn(global, 'setTimeout');
    const clearTimeoutSpy = jest.spyOn(global, 'clearTimeout');

    let settle: (value: Response) => void = () => {};
    fetchMock.mockImplementation((() => new Promise<Response>((resolve) => (settle = resolve))) as never);

    const { result } = renderHook(() => useServerInfo(), { wrapper });
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalled();
    });

    const abortTimerIndex = setTimeoutSpy.mock.calls.findIndex((call) => call[1] === SERVER_INFO_TIMEOUT_MS);
    expect(abortTimerIndex).toBeGreaterThanOrEqual(0);
    const abortTimer = setTimeoutSpy.mock.results[abortTimerIndex].value as ReturnType<typeof setTimeout>;

    jest.advanceTimersByTime(SERVER_INFO_TIMEOUT_MS - 1_000);
    settle(jsonResponse(SERVER_RESPONSE));

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });
    expect(result.current.data).toEqual(SERVER_RESPONSE);
    expect(clearTimeoutSpy).toHaveBeenCalledWith(abortTimer);

    // And the response survives past the deadline it would have aborted at.
    jest.advanceTimersByTime(2_000);
    expect(result.current.data).toEqual(SERVER_RESPONSE);

    setTimeoutSpy.mockRestore();
    clearTimeoutSpy.mockRestore();
  });
});
