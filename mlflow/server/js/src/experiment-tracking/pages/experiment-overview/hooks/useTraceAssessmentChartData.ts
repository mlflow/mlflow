import { useMemo, useCallback } from 'react';
import {
  MetricViewType,
  AggregationType,
  AssessmentMetricKey,
  AssessmentFilterKey,
  createAssessmentFilter,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { formatTimestampForTraceMetrics, useTimestampValueMap } from '../utils/chartUtils';
import type { OverviewChartProps } from '../types';

export interface AssessmentChartDataPoint {
  name: string;
  value: number;
}

export interface UseTraceAssessmentChartDataResult {
  /** Processed chart data with all time buckets filled */
  chartData: AssessmentChartDataPoint[];
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there are any data points */
  hasData: boolean;
}

export interface UseTraceAssessmentChartDataParams extends OverviewChartProps {
  /** The name of the assessment to fetch data for */
  assessmentName: string;
}

/**
 * Custom hook that fetches and processes assessment chart data.
 * Encapsulates all data-fetching and processing logic for individual assessment charts.
 *
 * @param props - Chart props including experimentId, time range, buckets, and assessment name
 * @returns Processed chart data, loading state, and error state
 */
export function useTraceAssessmentChartData({
  experimentId,
  startTimeMs,
  endTimeMs,
  timeIntervalSeconds,
  timeBuckets,
  assessmentName,
}: UseTraceAssessmentChartDataParams): UseTraceAssessmentChartDataResult {
  // Create filters for feedback assessments with the given name
  const filters = useMemo(() => [createAssessmentFilter(AssessmentFilterKey.NAME, assessmentName)], [assessmentName]);

  // Fetch assessment values over time
  const {
    data: timeSeriesData,
    isLoading,
    error,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.ASSESSMENTS,
    metricName: AssessmentMetricKey.ASSESSMENT_VALUE,
    aggregations: [{ aggregation_type: AggregationType.AVG }],
    filters,
    timeIntervalSeconds,
  });

  const timeSeriesDataPoints = useMemo(() => timeSeriesData?.data_points || [], [timeSeriesData?.data_points]);

  // Create a map of values by timestamp
  const valueExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => dp.values?.[AggregationType.AVG] || 0,
    [],
  );
  const valuesByTimestamp = useTimestampValueMap(timeSeriesDataPoints, valueExtractor);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => {
      const value = valuesByTimestamp.get(timestampMs);
      return {
        name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
        value: value || 0,
      };
    });
  }, [timeBuckets, valuesByTimestamp, timeIntervalSeconds]);

  return {
    chartData,
    isLoading,
    error,
    hasData: timeSeriesDataPoints.length > 0,
  };
}
