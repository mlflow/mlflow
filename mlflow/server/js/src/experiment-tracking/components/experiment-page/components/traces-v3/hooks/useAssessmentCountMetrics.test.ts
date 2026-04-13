import { jest, describe, it, expect } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { useAssessmentCountMetrics } from './useAssessmentCountMetrics';
import { AggregationType, AssessmentDimensionKey } from '@databricks/web-shared/model-trace-explorer';
import { shouldUseInfinitePaginatedTraces } from '@databricks/web-shared/genai-traces-table';

jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  shouldUseInfinitePaginatedTraces: jest.fn(() => true),
}));

const mockShouldUseInfinitePaginatedTraces = jest.mocked(shouldUseInfinitePaginatedTraces);

const mockUseTraceMetricsQuery = jest.fn();
jest.mock('../../../../../pages/experiment-overview/hooks/useTraceMetricsQuery', () => ({
  useTraceMetricsQuery: (...args: any[]) => mockUseTraceMetricsQuery(...args),
}));

describe('useAssessmentCountMetrics', () => {
  const defaultParams = {
    experimentIds: ['exp-1'],
    disabled: false,
  };

  it('transforms API response into assessment count metrics', () => {
    mockUseTraceMetricsQuery.mockReturnValue({
      data: {
        data_points: [
          {
            metric_name: 'assessment_count',
            dimensions: {
              [AssessmentDimensionKey.ASSESSMENT_NAME]: 'quality',
              [AssessmentDimensionKey.ASSESSMENT_VALUE]: '"yes"',
            },
            values: { [AggregationType.COUNT]: 50 },
          },
          {
            metric_name: 'assessment_count',
            dimensions: {
              [AssessmentDimensionKey.ASSESSMENT_NAME]: 'quality',
              [AssessmentDimensionKey.ASSESSMENT_VALUE]: '"no"',
            },
            values: { [AggregationType.COUNT]: 10 },
          },
        ],
      },
      isLoading: false,
    });

    const { result } = renderHook(() => useAssessmentCountMetrics(defaultParams));

    expect(result.current).toBeDefined();
    expect(result.current?.data).toEqual([
      { assessmentName: 'quality', assessmentValue: '"yes"', count: 50 },
      { assessmentName: 'quality', assessmentValue: '"no"', count: 10 },
    ]);
    expect(result.current?.isLoading).toBe(false);
  });

  it('returns undefined when feature flag is off', () => {
    mockShouldUseInfinitePaginatedTraces.mockReturnValue(false);

    mockUseTraceMetricsQuery.mockReturnValue({ data: undefined, isLoading: false });

    const { result } = renderHook(() => useAssessmentCountMetrics(defaultParams));
    expect(result.current).toBeUndefined();

    mockShouldUseInfinitePaginatedTraces.mockReturnValue(true);
  });

  it('returns undefined when disabled', () => {
    mockUseTraceMetricsQuery.mockReturnValue({ data: undefined, isLoading: false });

    const { result } = renderHook(() => useAssessmentCountMetrics({ ...defaultParams, disabled: true }));
    expect(result.current).toBeUndefined();
  });

  it('returns empty data when API returns no data points', () => {
    mockUseTraceMetricsQuery.mockReturnValue({ data: undefined, isLoading: false });

    const { result } = renderHook(() => useAssessmentCountMetrics(defaultParams));
    expect(result.current?.data).toEqual([]);
  });

  it('passes timeRange as milliseconds to the query', () => {
    mockUseTraceMetricsQuery.mockReturnValue({ data: undefined, isLoading: false });

    renderHook(() =>
      useAssessmentCountMetrics({
        ...defaultParams,
        timeRange: { startTime: '1000', endTime: '2000' },
      }),
    );

    expect(mockUseTraceMetricsQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        startTimeMs: 1000,
        endTimeMs: 2000,
      }),
    );
  });

  it('passes run UUID as metadata filter', () => {
    mockUseTraceMetricsQuery.mockReturnValue({ data: undefined, isLoading: false });

    renderHook(() =>
      useAssessmentCountMetrics({
        ...defaultParams,
        runUuid: 'run-123',
      }),
    );

    expect(mockUseTraceMetricsQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        filters: [expect.stringContaining('run-123')],
      }),
    );
  });
});
