import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';

import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { useGetTrace } from './useGetTrace';

describe('useGetTrace', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          // Don't override retry settings for tests
        },
      },
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  const mockTrace: ModelTrace = {
    data: {
      spans: [],
    },
    info: {
      request_id: 'test-request-id',
      timestamp_ms: 1234567890,
      execution_time_ms: 100,
      status: 'OK',
      request_metadata: [],
      tags: [],
    },
  };

  test('should be disabled when getTrace is not provided', () => {
    const mockGetTrace = jest.fn();
    const { result } = renderHook(() => useGetTrace(undefined, 'trace-id'), { wrapper });

    // Query should be disabled when getTrace is nil (enabled: !isNil(getTrace) && ...)
    // The enabled condition evaluates to false when getTrace is undefined
    expect(result.current.fetchStatus).toBe('idle');
    expect(result.current.isSuccess).toBe(false);
    expect(result.current.data).toBeUndefined();

    // Verify that no getTrace function was called since none was provided
    expect(mockGetTrace).not.toHaveBeenCalled();
  });

  test('should be disabled when traceId is not provided', () => {
    const mockGetTrace = jest.fn().mockResolvedValue(mockTrace);
    const { result } = renderHook(() => useGetTrace(mockGetTrace, undefined), { wrapper });

    // Query should be disabled when both requestId and traceId are nil
    // The enabled condition evaluates to false when both requestId and traceId are undefined
    expect(result.current.fetchStatus).toBe('idle');
    expect(result.current.isSuccess).toBe(false);
    expect(mockGetTrace).not.toHaveBeenCalled();
  });

  test('should fetch trace when getTrace and traceId are provided', async () => {
    const mockGetTrace = jest.fn().mockResolvedValue(mockTrace);
    const { result } = renderHook(() => useGetTrace(mockGetTrace, 'trace-id'), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(mockGetTrace).toHaveBeenCalledWith('trace-id');
    expect(result.current.data).toEqual(mockTrace);
  });
});
