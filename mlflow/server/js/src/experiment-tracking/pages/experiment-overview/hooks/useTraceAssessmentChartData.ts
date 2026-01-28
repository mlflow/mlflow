import { useMemo, useCallback } from 'react';
import {
  MetricViewType,
  AggregationType,
  AssessmentMetricKey,
  AssessmentFilterKey,
  AssessmentDimensionKey,
  createAssessmentFilter,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { formatTimestampForTraceMetrics, useTimestampValueMap } from '../utils/chartUtils';
import {
  sortValuesAlphanumerically,
  shouldCreateHistogramBuckets,
  createHistogramBuckets,
  findBucketIndexForValue,
} from '../utils/distributionUtils';
import { useOverviewChartContext } from '../OverviewChartContext';

export interface AssessmentChartDataPoint {
  name: string;
  value: number | null;
}

export interface DistributionChartDataPoint {
  name: string;
  count: number;
}

export interface UseTraceAssessmentChartDataResult {
  /** Processed time series chart data with all time buckets filled */
  timeSeriesChartData: AssessmentChartDataPoint[];
  /** Processed distribution chart data */
  distributionChartData: DistributionChartDataPoint[];
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there are any data points */
  hasData: boolean;
}

/**
 * Custom hook that fetches and processes assessment chart data.
 * Encapsulates all data-fetching and processing logic for individual assessment charts,
 * including both time series data and distribution data.
 * Uses OverviewChartContext to get chart props.
 *
 * @param assessmentName - The name of the assessment to fetch data for
 * @returns Processed chart data (time series and distribution), loading state, and error state
 */
export function useTraceAssessmentChartData(assessmentName: string): UseTraceAssessmentChartDataResult {
  const { experimentId, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets } = useOverviewChartContext();
  // Create filters for feedback assessments with the given name
  const filters = useMemo(() => [createAssessmentFilter(AssessmentFilterKey.NAME, assessmentName)], [assessmentName]);

  // Fetch assessment values over time for the line chart
  const {
    data: timeSeriesData,
    isLoading: isLoadingTimeSeries,
    error: timeSeriesError,
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

  // Fetch assessment counts grouped by assessment_value for the bar chart
  const {
    data: distributionData,
    isLoading: isLoadingDistribution,
    error: distributionError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.ASSESSMENTS,
    metricName: AssessmentMetricKey.ASSESSMENT_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    filters,
    dimensions: [AssessmentDimensionKey.ASSESSMENT_VALUE],
  });

  const timeSeriesDataPoints = useMemo(() => timeSeriesData?.data_points || [], [timeSeriesData?.data_points]);
  const distributionDataPoints = useMemo(() => distributionData?.data_points || [], [distributionData?.data_points]);

  // Create a map of values by timestamp for the line chart
  const valueExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => dp.values?.[AggregationType.AVG] ?? null,
    [],
  );
  const valuesByTimestamp = useTimestampValueMap(timeSeriesDataPoints, valueExtractor);

  // Prepare time series chart data - use null for missing data to show gaps in chart
  const timeSeriesChartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => ({
      name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
      value: valuesByTimestamp.get(timestampMs) ?? null,
    }));
  }, [timeBuckets, valuesByTimestamp, timeIntervalSeconds]);

  // Prepare distribution chart data - use actual values from API
  const distributionChartData = useMemo(() => {
    // Collect raw counts by assessment value
    const valueCounts: Record<string, number> = {};

    for (const dp of distributionDataPoints) {
      const rawValue = dp.dimensions?.[AssessmentDimensionKey.ASSESSMENT_VALUE];
      if (rawValue !== undefined) {
        const count = dp.values?.[AggregationType.COUNT] || 0;
        valueCounts[rawValue] = (valueCounts[rawValue] || 0) + count;
      }
    }

    const allValues = Object.keys(valueCounts);

    // Check if we should bucket into histogram ranges
    if (shouldCreateHistogramBuckets(allValues)) {
      const buckets = createHistogramBuckets(allValues);
      const bucketCounts = buckets.map(() => 0);

      // Aggregate counts into buckets
      for (const [value, count] of Object.entries(valueCounts)) {
        const numValue = parseFloat(value);
        if (!isNaN(numValue)) {
          const bucketIndex = findBucketIndexForValue(numValue, buckets);
          bucketCounts[bucketIndex] += count;
        }
      }

      return buckets.map((bucket, index) => ({
        name: bucket.label,
        count: bucketCounts[index],
      }));
    }

    // For non-bucketed values (sparse integers, strings, booleans), use as-is
    const sortedValues = sortValuesAlphanumerically(allValues);
    return sortedValues.map((value) => ({
      name: value,
      count: valueCounts[value] || 0,
    }));
  }, [distributionDataPoints]);

  const isLoading = isLoadingTimeSeries || isLoadingDistribution;
  const error = timeSeriesError || distributionError;
  const hasData = timeSeriesDataPoints.length > 0 || distributionDataPoints.length > 0;

  return {
    timeSeriesChartData,
    distributionChartData,
    isLoading,
    error,
    hasData,
  };
}
