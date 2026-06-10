import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { renderHook } from '@testing-library/react';

import { useCountInfo } from './useCountInfo';
import { AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import { shouldUseInfinitePaginatedTraces } from '@databricks/web-shared/genai-traces-table';
import { createTestTraceInfoV3 } from '@databricks/web-shared/genai-traces-table';

jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  ...jest.requireActual<typeof import('@databricks/web-shared/genai-traces-table')>(
    '@databricks/web-shared/genai-traces-table',
  ),
  shouldUseInfinitePaginatedTraces: jest.fn(() => false),
}));

const mockShouldUseInfinitePaginatedTraces = jest.mocked(shouldUseInfinitePaginatedTraces);

const mockUseTraceMetricsQuery = jest.fn();
jest.mock('../../../../../pages/experiment-overview/hooks/useTraceMetricsQuery', () => ({
  useTraceMetricsQuery: (...args: any[]) => mockUseTraceMetricsQuery(...args),
}));

describe('useCountInfo', () => {
  const defaultParams = {
    experimentIds: ['exp-1'],
    traceInfos: [],
    metadataTraceInfos: [],
    traceInfosLoading: false,
    metadataTotalCount: 0,
    disabled: false,
  };

  beforeEach(() => {
    mockShouldUseInfinitePaginatedTraces.mockReturnValue(false);
    mockUseTraceMetricsQuery.mockReset();
  });

  it('counts unique session ids when countSessions is enabled', () => {
    mockUseTraceMetricsQuery.mockReturnValue({
      data: {
        data_points: [{ values: { [AggregationType.COUNT]: 3 } }],
      },
      isLoading: false,
    });

    const traceInfos = [
      {
        ...createTestTraceInfoV3('tr-1', 'req-1', 'request-1'),
        trace_metadata: { 'mlflow.trace.session': 'session-a' },
      },
      {
        ...createTestTraceInfoV3('tr-2', 'req-2', 'request-2'),
        trace_metadata: { 'mlflow.trace.session': 'session-a' },
      },
      {
        ...createTestTraceInfoV3('tr-3', 'req-3', 'request-3'),
        trace_metadata: { 'mlflow.trace.session': 'session-b' },
      },
    ];

    const { result } = renderHook(() =>
      useCountInfo({
        ...defaultParams,
        traceInfos,
        metadataTraceInfos: traceInfos,
        metadataTotalCount: 3,
        countSessions: true,
      }),
    );

    expect(result.current).toEqual({
      currentCount: 2,
      logCountLoading: false,
      totalCount: 2,
      maxAllowedCount: 1000,
    });
    expect(mockUseTraceMetricsQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        metricName: TraceMetricKey.SESSION_COUNT,
        enabled: false,
      }),
    );
  });

  it('uses trace counts when not grouped by session', () => {
    mockUseTraceMetricsQuery.mockReturnValue({ data: undefined, isLoading: false });

    const traceInfos = [
      createTestTraceInfoV3('tr-1', 'req-1', 'request-1'),
      createTestTraceInfoV3('tr-2', 'req-2', 'request-2'),
    ];

    const { result } = renderHook(() =>
      useCountInfo({
        ...defaultParams,
        traceInfos,
        metadataTotalCount: 5,
      }),
    );

    expect(result.current).toEqual({
      currentCount: 2,
      logCountLoading: false,
      totalCount: 5,
      maxAllowedCount: 1000,
    });
    expect(mockUseTraceMetricsQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        metricName: TraceMetricKey.TRACE_COUNT,
        enabled: false,
      }),
    );
  });

  it('queries trace totals when infinite pagination is enabled', () => {
    mockShouldUseInfinitePaginatedTraces.mockReturnValue(true);
    mockUseTraceMetricsQuery.mockReturnValue({
      data: {
        data_points: [{ values: { [AggregationType.COUNT]: 7 } }],
      },
      isLoading: false,
    });

    const traceInfos = [
      createTestTraceInfoV3('tr-1', 'req-1', 'request-1'),
      createTestTraceInfoV3('tr-2', 'req-2', 'request-2'),
    ];

    const { result } = renderHook(() =>
      useCountInfo({
        ...defaultParams,
        traceInfos,
        metadataTotalCount: 2,
      }),
    );

    expect(result.current.totalCount).toBe(7);
    expect(result.current.currentCount).toBe(2);

    mockShouldUseInfinitePaginatedTraces.mockReturnValue(false);
  });

  it('falls back to traceInfosCount when traceInfos is unavailable', () => {
    mockUseTraceMetricsQuery.mockReturnValue({ data: undefined, isLoading: false });

    const { result } = renderHook(() =>
      useCountInfo({
        ...defaultParams,
        traceInfos: undefined,
        traceInfosCount: 4,
        metadataTotalCount: 9,
      }),
    );

    expect(result.current).toEqual({
      currentCount: 4,
      logCountLoading: false,
      totalCount: 9,
      maxAllowedCount: 1000,
    });
  });

  it('uses the session count metric when countSessions is enabled in infinite mode', () => {
    mockShouldUseInfinitePaginatedTraces.mockReturnValue(true);
    mockUseTraceMetricsQuery.mockReturnValue({ data: undefined, isLoading: false });

    const traceInfos = [
      {
        ...createTestTraceInfoV3('tr-1', 'req-1', 'request-1'),
        trace_metadata: { 'mlflow.trace.session': 'session-a' },
      },
    ];

    renderHook(() =>
      useCountInfo({
        ...defaultParams,
        runUuid: 'run-123',
        traceInfos,
        metadataTraceInfos: traceInfos,
        countSessions: true,
      }),
    );

    expect(mockUseTraceMetricsQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        metricName: TraceMetricKey.SESSION_COUNT,
        enabled: true,
        filters: ['trace.metadata.`mlflow.sourceRun` = "run-123"'],
      }),
    );
  });

  it('falls back to loaded trace counts in infinite mode when metrics are unavailable', () => {
    mockShouldUseInfinitePaginatedTraces.mockReturnValue(true);
    mockUseTraceMetricsQuery.mockReturnValue({ data: undefined, isLoading: false });

    const traceInfos = [
      createTestTraceInfoV3('tr-1', 'req-1', 'request-1'),
      createTestTraceInfoV3('tr-2', 'req-2', 'request-2'),
    ];

    const { result } = renderHook(() =>
      useCountInfo({
        ...defaultParams,
        traceInfos,
        metadataTotalCount: 1,
      }),
    );

    expect(result.current).toEqual({
      currentCount: 2,
      logCountLoading: false,
      totalCount: 2,
      maxAllowedCount: Infinity,
    });
  });
});
