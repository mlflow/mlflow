import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import React from 'react';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '../../common/utils/reactQueryHooks';
import { useTryIt } from './useTryIt';
import { fetchOrFail } from '../../common/utils/FetchUtils';
import { GenericNetworkRequestError } from '../../shared/web-shared/errors/PredefinedErrors';

jest.mock('../../common/utils/FetchUtils', () => ({
  fetchOrFail: jest.fn(),
}));

const mockFetchOrFail = jest.mocked(fetchOrFail);

const TRY_IT_URL = 'http://test/gateway/endpoint/mlflow/invocations';
const validRequestBody = JSON.stringify({ inputs: ['hello'] });

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      mutations: {
        retry: false,
      },
    },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  };
}

describe('useTryIt', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('returns formatted response on success', async () => {
    const mockPayload = { output: 'Hello from model' };
    mockFetchOrFail.mockResolvedValueOnce({
      text: () => Promise.resolve(JSON.stringify(mockPayload)),
    } as Response);

    const { result } = renderHook(() => useTryIt({ tryItRequestUrl: TRY_IT_URL }), {
      wrapper: createWrapper(),
    });

    expect(result.current.data).toBeUndefined();
    expect(result.current.error).toBeUndefined();
    expect(result.current.isLoading).toBe(false);

    result.current.sendRequest(validRequestBody);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(mockFetchOrFail).toHaveBeenCalledTimes(1);
    expect(mockFetchOrFail).toHaveBeenCalledWith(TRY_IT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: validRequestBody,
    });
    expect(result.current.data).toBe(JSON.stringify(mockPayload, null, 2));
    expect(result.current.error).toBeUndefined();
  });

  test('sets error with response body when request fails with NetworkRequestError', async () => {
    const errorBody = { detail: 'The request was invalid.' };
    const mockResponse = {
      text: () => Promise.resolve(JSON.stringify(errorBody)),
    } as unknown as Response;
    const networkError = new GenericNetworkRequestError({ status: 400, response: mockResponse });
    mockFetchOrFail.mockRejectedValueOnce(networkError);

    const { result } = renderHook(() => useTryIt({ tryItRequestUrl: TRY_IT_URL }), {
      wrapper: createWrapper(),
    });

    result.current.sendRequest(validRequestBody);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(mockFetchOrFail).toHaveBeenCalledTimes(1);
    expect(result.current.data).toBeUndefined();
    expect(result.current.error).toBeDefined();
    expect(result.current.error?.message).toBe('A network error occurred.');
    expect(result.current.error?.responseBody).toBe(JSON.stringify(errorBody, null, 2));
  });

  test('sets error without response body when request fails with non-NetworkRequestError', async () => {
    mockFetchOrFail.mockRejectedValueOnce(new Error('Network failure'));

    const { result } = renderHook(() => useTryIt({ tryItRequestUrl: TRY_IT_URL }), {
      wrapper: createWrapper(),
    });

    result.current.sendRequest(validRequestBody);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeDefined();
    expect(result.current.error?.message).toBe('Network failure');
    expect(result.current.error?.responseBody).toBeUndefined();
  });

  test('sets error and does not call fetch when request body is invalid JSON', async () => {
    const { result } = renderHook(() => useTryIt({ tryItRequestUrl: TRY_IT_URL }), {
      wrapper: createWrapper(),
    });

    result.current.sendRequest('not valid json');

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(mockFetchOrFail).not.toHaveBeenCalled();
    expect(result.current.error).toBeDefined();
    expect(result.current.error?.message).toBe('Invalid JSON in request body');
    expect(result.current.error?.responseBody).toBeUndefined();
  });

  test('reset clears data and error', async () => {
    mockFetchOrFail.mockResolvedValueOnce({
      text: () => Promise.resolve(JSON.stringify({ ok: true })),
    } as Response);

    const { result } = renderHook(() => useTryIt({ tryItRequestUrl: TRY_IT_URL }), {
      wrapper: createWrapper(),
    });

    result.current.sendRequest(validRequestBody);

    await waitFor(() => {
      expect(result.current.data).toBeDefined();
    });

    result.current.reset();

    await waitFor(() => {
      expect(result.current.data).toBeUndefined();
      expect(result.current.error).toBeUndefined();
    });
  });

  test('reset clears error state after failed request', async () => {
    mockFetchOrFail.mockRejectedValueOnce(new Error('Server error'));

    const { result } = renderHook(() => useTryIt({ tryItRequestUrl: TRY_IT_URL }), {
      wrapper: createWrapper(),
    });

    result.current.sendRequest(validRequestBody);

    await waitFor(() => {
      expect(result.current.error).toBeDefined();
    });

    result.current.reset();

    await waitFor(() => {
      expect(result.current.error).toBeUndefined();
    });
  });
});
