import { describe, beforeEach, afterEach, jest, test, expect } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';

import type { ModelTrace, ModelTraceInfoV3 } from '../../model-trace-explorer/ModelTrace.types';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

import type { GetTraceFunction } from './useGetTrace';
import { useGetTrace } from './useGetTrace';

describe('useGetTrace', () => {
  let queryClient: QueryClient;

  const demoTraceInfo: ModelTraceInfoV3 = {
    trace_id: 'trace-id-123',
    trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'exp-1' } },
    request_time: '1625247600000',
    state: 'OK',
    trace_metadata: {},
    tags: {},
  };

  beforeEach(() => {
    jest.useFakeTimers();
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          // Don't override retry settings for tests
        },
      },
    });
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
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
    const { result } = renderHook(() => useGetTrace(undefined, demoTraceInfo), { wrapper });

    // Query should be disabled when getTrace is nil (enabled: !isNil(getTrace) && ...)
    // The enabled condition evaluates to false when getTrace is undefined
    expect(result.current.fetchStatus).toBe('idle');
    expect(result.current.isSuccess).toBe(false);
    expect(result.current.data).toBeUndefined();

    // Verify that no getTrace function was called since none was provided
    expect(mockGetTrace).not.toHaveBeenCalled();
  });

  test('should be disabled when traceId is not provided', () => {
    const mockGetTrace = jest.fn<GetTraceFunction>().mockResolvedValue(mockTrace);
    const { result } = renderHook(() => useGetTrace(mockGetTrace, undefined), { wrapper });

    // Query should be disabled when both requestId and traceId are nil
    // The enabled condition evaluates to false when both requestId and traceId are undefined
    expect(result.current.fetchStatus).toBe('idle');
    expect(result.current.isSuccess).toBe(false);
    expect(mockGetTrace).not.toHaveBeenCalled();
  });

  test('should fetch trace when getTrace and traceId are provided', async () => {
    const mockGetTrace = jest.fn<GetTraceFunction>().mockResolvedValue(mockTrace);
    const { result } = renderHook(() => useGetTrace(mockGetTrace, demoTraceInfo), { wrapper });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    // prettier-ignore
    expect(mockGetTrace).toHaveBeenCalledWith(
      'trace-id-123',
      demoTraceInfo,
    );
    expect(result.current.data).toEqual(mockTrace);
  });

  describe('polling behavior', () => {
    test('should stop polling after max retries when OK state has span count mismatch', async () => {
      const traceWithOKState: ModelTrace = {
        data: { spans: [{ name: 'span1' }] as any },
        info: {
          trace_id: 'trace-id-123',
          trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'exp-1' } },
          request_time: '1625247600000',
          state: 'OK',
          trace_metadata: {
            'mlflow.trace.sizeStats': JSON.stringify({ num_spans: 5 }), // Mismatch: metadata says 5, but only 1 span
          },
          tags: {},
        } as ModelTraceInfoV3,
      };

      const mockGetTrace = jest.fn<GetTraceFunction>().mockResolvedValue(traceWithOKState);
      const { result } = renderHook(() => useGetTrace(mockGetTrace, traceWithOKState.info, true), { wrapper });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // Advance timers to simulate polling attempts
      jest.advanceTimersByTime(3000);

      await waitFor(() => {
        const callCount = mockGetTrace.mock.calls.length;
        // Should have polled more than once (initial + retries) but not indefinitely
        expect(callCount).toBeGreaterThan(1);
        expect(callCount).toBeLessThanOrEqual(61); // 1 initial + max 60 retries
      });
    });

    test('should reset poll counter when traceId changes', async () => {
      const traceInfoA: ModelTraceInfoV3 = {
        trace_id: 'trace-A',
        trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'exp-1' } },
        request_time: '1625247600000',
        state: 'OK',
        trace_metadata: {
          'mlflow.trace.sizeStats': JSON.stringify({ num_spans: 5 }),
        },
        tags: {},
      };

      const traceInfoB: ModelTraceInfoV3 = {
        trace_id: 'trace-B',
        trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'exp-1' } },
        request_time: '1625247600000',
        state: 'OK',
        trace_metadata: {
          'mlflow.trace.sizeStats': JSON.stringify({ num_spans: 5 }),
        },
        tags: {},
      };

      const traceWithMismatchA: ModelTrace = {
        data: { spans: [{ name: 'span1' }] as any },
        info: traceInfoA as ModelTraceInfoV3,
      };
      const traceWithMismatchB: ModelTrace = {
        data: { spans: [{ name: 'span1' }] as any },
        info: traceInfoB as ModelTraceInfoV3,
      };

      const mockGetTrace = jest.fn<GetTraceFunction>().mockImplementation((traceId) => {
        return Promise.resolve(traceId === 'trace-A' ? traceWithMismatchA : traceWithMismatchB);
      });

      let currentTraceInfo = traceInfoA;
      const { result, rerender } = renderHook(() => useGetTrace(mockGetTrace, currentTraceInfo, true), { wrapper });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // Advance timers to let trace A poll for a bit
      jest.advanceTimersByTime(2000);

      await waitFor(() => {
        const callsForTraceA = mockGetTrace.mock.calls.filter((c: any) => c[0] === 'trace-A').length;
        expect(callsForTraceA).toBeGreaterThan(1);
      });

      // Switch to trace B - poll counter should reset
      currentTraceInfo = traceInfoB;
      rerender();

      await waitFor(() => {
        expect(mockGetTrace).toHaveBeenCalledWith('trace-B', traceInfoB);
      });

      // Advance timers to let trace B poll for a bit
      jest.advanceTimersByTime(2000);

      await waitFor(() => {
        const callsForTraceB = mockGetTrace.mock.calls.filter((c: any) => c[0] === 'trace-B').length;
        // Trace B should get its own full set of polling attempts (not reduced by trace A's count)
        expect(callsForTraceB).toBeGreaterThan(1);
      });
    });

    test('should not poll when trace state is ERROR', async () => {
      const traceWithErrorState: ModelTrace = {
        data: { spans: [] },
        info: {
          trace_id: 'trace-id-123',
          trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'exp-1' } },
          request_time: '1625247600000',
          state: 'ERROR',
          trace_metadata: {},
          tags: {},
        } as ModelTraceInfoV3,
      };

      const mockGetTrace = jest.fn<GetTraceFunction>().mockResolvedValue(traceWithErrorState);
      const { result } = renderHook(() => useGetTrace(mockGetTrace, traceWithErrorState.info, true), { wrapper });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // Advance timers to ensure no additional calls are made
      jest.advanceTimersByTime(1500);

      expect(mockGetTrace).toHaveBeenCalledTimes(1);
    });
  });

  describe('polling with spans_complete flag (V3 OSS backend)', () => {
    const okTraceInfo: ModelTraceInfoV3 = {
      trace_id: 'trace-id-123',
      trace_location: { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'exp-1' } },
      request_time: '1625247600000',
      state: 'OK',
      trace_metadata: {},
      tags: {},
    };

    test('should stop polling immediately when spans_complete is true', async () => {
      const completeTrace: ModelTrace = {
        data: { spans: [{ name: 'span1' }] as any },
        info: okTraceInfo,
        spans_complete: true,
      };

      const mockGetTrace = jest.fn<GetTraceFunction>().mockResolvedValue(completeTrace);
      const { result } = renderHook(() => useGetTrace(mockGetTrace, okTraceInfo, true), { wrapper });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // Advance timers — should NOT trigger additional fetches
      jest.advanceTimersByTime(3000);

      // Only the initial fetch; polling should have stopped immediately
      expect(mockGetTrace).toHaveBeenCalledTimes(1);
    });

    test('should continue polling when spans_complete is false', async () => {
      const incompleteTrace: ModelTrace = {
        data: { spans: [{ name: 'span1' }] as any },
        info: okTraceInfo,
        spans_complete: false,
      };

      const mockGetTrace = jest.fn<GetTraceFunction>().mockResolvedValue(incompleteTrace);
      const { result } = renderHook(() => useGetTrace(mockGetTrace, okTraceInfo, true), { wrapper });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      jest.advanceTimersByTime(3000);

      await waitFor(() => {
        expect(mockGetTrace.mock.calls.length).toBeGreaterThan(1);
      });
    });

    test('should stop after max retries when spans_complete stays false', async () => {
      const incompleteTrace: ModelTrace = {
        data: { spans: [{ name: 'span1' }] as any },
        info: okTraceInfo,
        spans_complete: false,
      };

      const mockGetTrace = jest.fn<GetTraceFunction>().mockResolvedValue(incompleteTrace);
      const { result } = renderHook(() => useGetTrace(mockGetTrace, okTraceInfo, true), { wrapper });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      jest.advanceTimersByTime(65000);

      await waitFor(() => {
        const callCount = mockGetTrace.mock.calls.length;
        expect(callCount).toBeGreaterThan(1);
        expect(callCount).toBeLessThanOrEqual(61); // 1 initial + max 60 retries
      });
    });

    test('should use manual span-count fallback when spans_complete is undefined', async () => {
      // Simulates an older backend that does not return spans_complete.
      // The hook should fall back to manual num_spans comparison.
      const traceWithMatchingSpans: ModelTrace = {
        data: { spans: [{ name: 'span1' }] as any },
        info: {
          ...okTraceInfo,
          trace_metadata: {
            'mlflow.trace.sizeStats': JSON.stringify({ num_spans: 1 }),
          },
        } as ModelTraceInfoV3,
        // spans_complete is absent (undefined)
      };

      const mockGetTrace = jest.fn<GetTraceFunction>().mockResolvedValue(traceWithMatchingSpans);
      const { result } = renderHook(
        () => useGetTrace(mockGetTrace, traceWithMatchingSpans.info, true),
        { wrapper },
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      jest.advanceTimersByTime(3000);

      // span count matches num_spans → polling stops after initial fetch
      expect(mockGetTrace).toHaveBeenCalledTimes(1);
    });
  });
});
