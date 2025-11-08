import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { useDeleteTracesMutation } from './useDeleteTraces';

// Mock the dependencies
jest.mock('../../../sdk/MlflowService', () => ({
  MlflowService: {
    deleteTracesV3: jest.fn(),
  },
}));

jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  invalidateMlflowSearchTracesCache: jest.fn(),
}));

// Import the mocked dependencies
import { MlflowService } from '../../../sdk/MlflowService';
import { invalidateMlflowSearchTracesCache } from '@databricks/web-shared/genai-traces-table';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false, // Turn off retries to simplify testing
      },
    },
    logger: {
      error: () => {},
      log: () => {},
      warn: () => {},
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe('useDeleteTracesMutation', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('should call MlflowService.deleteTracesV3 with correct parameters for single batch', async () => {
    const mockResponse = { traces_deleted: 5 };
    jest.mocked(MlflowService.deleteTracesV3).mockResolvedValueOnce(mockResponse);

    const { result } = renderHook(() => useDeleteTracesMutation(), {
      wrapper: createWrapper(),
    });

    const traceIds = ['trace-1', 'trace-2', 'trace-3', 'trace-4', 'trace-5'];
    result.current.mutate({
      experimentId: 'test-experiment-id',
      traceRequestIds: traceIds,
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    // Should make single API call since we have less than 100 traces
    expect(MlflowService.deleteTracesV3).toHaveBeenCalledTimes(1);
    expect(MlflowService.deleteTracesV3).toHaveBeenCalledWith('test-experiment-id', traceIds);

    // Should return the correct result
    expect(result.current.data).toEqual({ traces_deleted: 5 });

    // Should invalidate cache on success
    expect(invalidateMlflowSearchTracesCache).toHaveBeenCalledTimes(1);
    expect(invalidateMlflowSearchTracesCache).toHaveBeenCalledWith({ queryClient: expect.any(QueryClient) });
  });

  test('should batch trace IDs into chunks of 100 and process in parallel', async () => {
    // Create 250 trace IDs to test batching
    const traceIds = Array.from({ length: 250 }, (_, i) => `trace-${i + 1}`);

    // Mock responses for each batch
    const mockResponses = [
      { traces_deleted: 100 }, // First batch: 100 traces
      { traces_deleted: 100 }, // Second batch: 100 traces
      { traces_deleted: 50 }, // Third batch: 50 traces
    ];

    jest
      .mocked(MlflowService.deleteTracesV3)
      .mockResolvedValueOnce(mockResponses[0])
      .mockResolvedValueOnce(mockResponses[1])
      .mockResolvedValueOnce(mockResponses[2]);

    const { result } = renderHook(() => useDeleteTracesMutation(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({
      experimentId: 'test-experiment-id',
      traceRequestIds: traceIds,
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    // Should make 3 API calls (250 traces / 100 per batch = 3 batches)
    expect(MlflowService.deleteTracesV3).toHaveBeenCalledTimes(3);

    // Verify first batch (traces 1-100)
    expect(MlflowService.deleteTracesV3).toHaveBeenNthCalledWith(1, 'test-experiment-id', traceIds.slice(0, 100));

    // Verify second batch (traces 101-200)
    expect(MlflowService.deleteTracesV3).toHaveBeenNthCalledWith(2, 'test-experiment-id', traceIds.slice(100, 200));

    // Verify third batch (traces 201-250)
    expect(MlflowService.deleteTracesV3).toHaveBeenNthCalledWith(3, 'test-experiment-id', traceIds.slice(200, 250));

    // Should sum up the total traces deleted
    expect(result.current.data).toEqual({ traces_deleted: 250 });

    // Should invalidate cache on success
    expect(invalidateMlflowSearchTracesCache).toHaveBeenCalledTimes(1);
  });

  test('should handle exactly 100 trace IDs without over-batching', async () => {
    const traceIds = Array.from({ length: 100 }, (_, i) => `trace-${i + 1}`);
    const mockResponse = { traces_deleted: 100 };

    jest.mocked(MlflowService.deleteTracesV3).mockResolvedValueOnce(mockResponse);

    const { result } = renderHook(() => useDeleteTracesMutation(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({
      experimentId: 'test-experiment-id',
      traceRequestIds: traceIds,
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    // Should make exactly 1 API call for 100 traces
    expect(MlflowService.deleteTracesV3).toHaveBeenCalledTimes(1);
    expect(MlflowService.deleteTracesV3).toHaveBeenCalledWith('test-experiment-id', traceIds);

    expect(result.current.data).toEqual({ traces_deleted: 100 });
  });

  test('should handle exactly 101 trace IDs by creating 2 batches', async () => {
    const traceIds = Array.from({ length: 101 }, (_, i) => `trace-${i + 1}`);

    const mockResponses = [
      { traces_deleted: 100 }, // First batch: 100 traces
      { traces_deleted: 1 }, // Second batch: 1 trace
    ];

    jest
      .mocked(MlflowService.deleteTracesV3)
      .mockResolvedValueOnce(mockResponses[0])
      .mockResolvedValueOnce(mockResponses[1]);

    const { result } = renderHook(() => useDeleteTracesMutation(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({
      experimentId: 'test-experiment-id',
      traceRequestIds: traceIds,
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    // Should make 2 API calls
    expect(MlflowService.deleteTracesV3).toHaveBeenCalledTimes(2);

    // Verify first batch (100 traces)
    expect(MlflowService.deleteTracesV3).toHaveBeenNthCalledWith(1, 'test-experiment-id', traceIds.slice(0, 100));

    // Verify second batch (1 trace)
    expect(MlflowService.deleteTracesV3).toHaveBeenNthCalledWith(2, 'test-experiment-id', traceIds.slice(100, 101));

    expect(result.current.data).toEqual({ traces_deleted: 101 });
  });

  test('should handle empty trace IDs array', async () => {
    const { result } = renderHook(() => useDeleteTracesMutation(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({
      experimentId: 'test-experiment-id',
      traceRequestIds: [],
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    // Should not make any API calls for empty array
    expect(MlflowService.deleteTracesV3).toHaveBeenCalledTimes(0);

    // Should return 0 traces deleted
    expect(result.current.data).toEqual({ traces_deleted: 0 });

    // Should still invalidate cache
    expect(invalidateMlflowSearchTracesCache).toHaveBeenCalledTimes(1);
  });

  test('should handle partial failures in batched requests', async () => {
    const traceIds = Array.from({ length: 150 }, (_, i) => `trace-${i + 1}`);

    // First batch succeeds, second batch fails
    jest
      .mocked(MlflowService.deleteTracesV3)
      .mockResolvedValueOnce({ traces_deleted: 100 })
      .mockRejectedValueOnce(new Error('Network error on second batch'));

    const { result } = renderHook(() => useDeleteTracesMutation(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({
      experimentId: 'test-experiment-id',
      traceRequestIds: traceIds,
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    // Should have attempted 2 API calls
    expect(MlflowService.deleteTracesV3).toHaveBeenCalledTimes(2);

    // Error should be from the failed batch
    expect(result.current.error).toBeInstanceOf(Error);
    expect((result.current.error as Error).message).toBe('Network error on second batch');

    // Should not invalidate cache on error
    expect(invalidateMlflowSearchTracesCache).toHaveBeenCalledTimes(0);
  });

  test('should handle single batch failure', async () => {
    const traceIds = ['trace-1', 'trace-2'];

    jest.mocked(MlflowService.deleteTracesV3).mockRejectedValueOnce(new Error('Permission denied'));

    const { result } = renderHook(() => useDeleteTracesMutation(), {
      wrapper: createWrapper(),
    });

    result.current.mutate({
      experimentId: 'test-experiment-id',
      traceRequestIds: traceIds,
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(MlflowService.deleteTracesV3).toHaveBeenCalledTimes(1);
    expect(result.current.error).toBeInstanceOf(Error);
    expect((result.current.error as Error).message).toBe('Permission denied');

    // Should not invalidate cache on error
    expect(invalidateMlflowSearchTracesCache).toHaveBeenCalledTimes(0);
  });

  test('should process all batches in parallel, not sequentially', async () => {
    const traceIds = Array.from({ length: 300 }, (_, i) => `trace-${i + 1}`);

    // Create promises that we can control timing of
    let resolveFirst: (value: any) => void;
    let resolveSecond: (value: any) => void;
    let resolveThird: (value: any) => void;

    const firstPromise = new Promise((resolve) => {
      resolveFirst = resolve;
    });
    const secondPromise = new Promise((resolve) => {
      resolveSecond = resolve;
    });
    const thirdPromise = new Promise((resolve) => {
      resolveThird = resolve;
    });

    jest
      .mocked(MlflowService.deleteTracesV3)
      .mockReturnValueOnce(firstPromise as any)
      .mockReturnValueOnce(secondPromise as any)
      .mockReturnValueOnce(thirdPromise as any);

    const { result } = renderHook(() => useDeleteTracesMutation(), {
      wrapper: createWrapper(),
    });

    // Start the mutation
    result.current.mutate({
      experimentId: 'test-experiment-id',
      traceRequestIds: traceIds,
    });

    // All three calls should be made immediately (parallel execution)
    await waitFor(() => {
      expect(MlflowService.deleteTracesV3).toHaveBeenCalledTimes(3);
    });

    // Resolve the promises in reverse order to verify they're parallel
    resolveThird!({ traces_deleted: 100 });
    resolveFirst!({ traces_deleted: 100 });
    resolveSecond!({ traces_deleted: 100 });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toEqual({ traces_deleted: 300 });
  });
});
