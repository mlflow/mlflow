import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { renderHook } from '@testing-library/react';

import { useTracesV4TraceCount } from './useTracesV4TraceCount';
import { AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';

const mockUseTraceMetricsQuery = jest.fn();
jest.mock('../../../../../pages/experiment-overview/hooks/useTraceMetricsQuery', () => ({
  useTraceMetricsQuery: (...args: any[]) => mockUseTraceMetricsQuery(...args),
}));

const timeRange = { startTime: '1000', endTime: '2000' };

describe('useTracesV4TraceCount', () => {
  beforeEach(() => {
    mockUseTraceMetricsQuery.mockReset();
    mockUseTraceMetricsQuery.mockReturnValue({ data: undefined, isLoading: false, isFetching: false });
  });

  it('uses the time-scoped metrics total on the normal (non-trace-id) path', () => {
    mockUseTraceMetricsQuery.mockReturnValue({
      data: { data_points: [{ values: { [AggregationType.COUNT]: 42 } }] },
      isLoading: false,
      isFetching: false,
    });

    const { result } = renderHook(() => useTracesV4TraceCount('exp-1', 10, timeRange));

    expect(result.current).toEqual({ currentCount: 10, totalCount: 42, isTotalLoading: false });
    // The metrics query is enabled when the search isn't an exact trace-id; sessions aren't counted.
    expect(mockUseTraceMetricsQuery).toHaveBeenCalledWith(
      expect.objectContaining({ enabled: true, metricName: TraceMetricKey.TRACE_COUNT }),
    );
    expect(mockUseTraceMetricsQuery).toHaveBeenCalledWith(
      expect.objectContaining({ enabled: false, metricName: TraceMetricKey.SESSION_COUNT }),
    );
  });

  it('also counts sessions from the session_count metric, scoped to the same time range', () => {
    mockUseTraceMetricsQuery.mockImplementation(({ metricName }: any) => ({
      data: {
        data_points: [{ values: { [AggregationType.COUNT]: metricName === TraceMetricKey.SESSION_COUNT ? 60 : 524 } }],
      },
      isLoading: false,
      isFetching: false,
    }));

    const { result } = renderHook(() => useTracesV4TraceCount('exp-1', 500, timeRange, { pageSessionCount: 58 }));

    expect(result.current).toEqual({
      currentCount: 500,
      totalCount: 524,
      isTotalLoading: false,
      sessions: { currentCount: 58, totalCount: 60 },
    });
    for (const metricName of [TraceMetricKey.TRACE_COUNT, TraceMetricKey.SESSION_COUNT]) {
      expect(mockUseTraceMetricsQuery).toHaveBeenCalledWith(
        expect.objectContaining({
          experimentIds: ['exp-1'],
          metricName,
          startTimeMs: 1000,
          endTimeMs: 2000,
          enabled: true,
        }),
      );
    }
  });

  it('keeps the totals loading until the session total resolves too', () => {
    mockUseTraceMetricsQuery.mockImplementation(({ metricName }: any) =>
      metricName === TraceMetricKey.SESSION_COUNT
        ? { data: undefined, isLoading: true, isFetching: true }
        : {
            data: { data_points: [{ values: { [AggregationType.COUNT]: 524 } }] },
            isLoading: false,
            isFetching: false,
          },
    );

    const { result } = renderHook(() => useTracesV4TraceCount('exp-1', 500, timeRange, { pageSessionCount: 58 }));

    expect(result.current.isTotalLoading).toBe(true);
    expect(result.current.sessions).toEqual({ currentCount: 58, totalCount: undefined });
  });

  it('spins (undefined total) while the metrics query is fetching, even with cached data', () => {
    // A re-enabled query serves prior cached data while re-fetching (e.g. leaving trace-id mode);
    // gating on isFetching avoids briefly surfacing that stale, differently-scoped total.
    mockUseTraceMetricsQuery.mockReturnValue({
      data: { data_points: [{ values: { [AggregationType.COUNT]: 99 } }] },
      isLoading: false,
      isFetching: true,
    });

    const { result } = renderHook(() => useTracesV4TraceCount('exp-1', 10, timeRange));

    expect(result.current).toEqual({ currentCount: 10, totalCount: undefined, isTotalLoading: true });
  });

  it('reports the resolved row count as the total for an exact trace-id search', () => {
    const { result } = renderHook(() => useTracesV4TraceCount('exp-1', 1, timeRange, { isExactTraceIdSearch: true }));

    expect(result.current).toEqual({ currentCount: 1, totalCount: 1, isTotalLoading: false });
    // The time-scoped metrics query is skipped entirely for a trace-id search.
    expect(mockUseTraceMetricsQuery).toHaveBeenCalledWith(expect.objectContaining({ enabled: false }));
  });

  it('keeps the total loading (undefined) while the exact-id result is still resolving', () => {
    // currentPageCount is the previous page's retained rows until the row query settles, so the
    // total must not be reported yet — otherwise the footer flashes the stale previous count.
    const { result } = renderHook(() =>
      useTracesV4TraceCount('exp-1', 50, timeRange, { isExactTraceIdSearch: true, isResultLoading: true }),
    );

    expect(result.current).toEqual({ currentCount: 50, totalCount: undefined, isTotalLoading: true });
  });

  it('reports the resolved sessions as the session total for an exact trace-id search', () => {
    // A matched trace with no session resolves to zero sessions, so the footer shows the trace alone.
    const { result } = renderHook(() =>
      useTracesV4TraceCount('exp-1', 1, timeRange, { isExactTraceIdSearch: true, pageSessionCount: 0 }),
    );

    expect(result.current).toEqual({
      currentCount: 1,
      totalCount: 1,
      isTotalLoading: false,
      sessions: { currentCount: 0, totalCount: 0 },
    });
    expect(mockUseTraceMetricsQuery).toHaveBeenCalledWith(
      expect.objectContaining({ enabled: false, metricName: TraceMetricKey.SESSION_COUNT }),
    );
  });
});
