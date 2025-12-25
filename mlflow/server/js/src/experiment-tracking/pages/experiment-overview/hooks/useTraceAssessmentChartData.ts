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
import type { OverviewChartProps } from '../types';

export interface AssessmentChartDataPoint {
  name: string;
  value: number;
}

export interface DistributionChartDataPoint {
  name: string;
  count: number;
}

export interface UseTraceAssessmentChartDataResult {
  /** Processed time series chart data with all time buckets filled */
  chartData: AssessmentChartDataPoint[];
  /** Processed distribution chart data */
  distributionChartData: DistributionChartDataPoint[];
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
 * Sort assessment values intelligently.
 * Numbers are sorted numerically, strings alphabetically.
 */
const sortAssessmentValues = (values: string[]): string[] =>
  [...values].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

/**
 * Determine if values should be bucketed into ranges.
 * Returns true if:
 * - Values are floats (have decimals), OR
 * - Values are integers with more than 5 unique values
 */
const shouldBucketValues = (values: string[]): boolean => {
  if (values.length === 0) return false;

  const numericValues = values.map((v) => parseFloat(v)).filter((n) => !isNaN(n));
  // Most values should be numeric
  if (numericValues.length < values.length * 0.8) return false;

  const uniqueValues = new Set(numericValues);
  const hasDecimals = numericValues.some((n) => !Number.isInteger(n));

  // Bucket if: values are floats, OR integers with more than 5 unique values
  return hasDecimals || uniqueValues.size > 5;
};

/**
 * Create buckets for continuous numeric values.
 * Returns bucket definitions based on min/max of the data.
 */
const createBuckets = (values: string[], numBuckets = 5): { min: number; max: number; label: string }[] => {
  const numericValues = values.map((v) => parseFloat(v)).filter((n) => !isNaN(n));
  if (numericValues.length === 0) return [];

  const min = Math.min(...numericValues);
  const max = Math.max(...numericValues);
  const range = max - min;
  const bucketSize = range / numBuckets;

  return Array.from({ length: numBuckets }, (_, i) => {
    const bucketMin = min + i * bucketSize;
    const bucketMax = i === numBuckets - 1 ? max : min + (i + 1) * bucketSize;
    return {
      min: bucketMin,
      max: bucketMax,
      label: `${bucketMin.toFixed(2)}-${bucketMax.toFixed(2)}`,
    };
  });
};

/**
 * Get the bucket index for a value.
 */
const getBucketIndex = (value: number, buckets: { min: number; max: number }[]): number => {
  for (let i = 0; i < buckets.length; i++) {
    if (value >= buckets[i].min && value <= buckets[i].max) {
      return i;
    }
  }
  return buckets.length - 1; // Default to last bucket
};

/**
 * Custom hook that fetches and processes assessment chart data.
 * Encapsulates all data-fetching and processing logic for individual assessment charts,
 * including both time series data and distribution data.
 *
 * @param props - Chart props including experimentId, time range, buckets, and assessment name
 * @returns Processed chart data (time series and distribution), loading state, and error state
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
    (dp: { values?: Record<string, number> }) => dp.values?.[AggregationType.AVG] || 0,
    [],
  );
  const valuesByTimestamp = useTimestampValueMap(timeSeriesDataPoints, valueExtractor);

  // Prepare time series chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => {
      const value = valuesByTimestamp.get(timestampMs);
      return {
        name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
        value: value || 0,
      };
    });
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

    // Check if we should bucket float values
    if (shouldBucketValues(allValues)) {
      const buckets = createBuckets(allValues);
      const bucketCounts = buckets.map(() => 0);

      // Aggregate counts into buckets
      for (const [value, count] of Object.entries(valueCounts)) {
        const numValue = parseFloat(value);
        if (!isNaN(numValue)) {
          const bucketIndex = getBucketIndex(numValue, buckets);
          bucketCounts[bucketIndex] += count;
        }
      }

      return buckets.map((bucket, index) => ({
        name: bucket.label,
        count: bucketCounts[index],
      }));
    }

    // For non-float values (integers, strings, booleans), use as-is
    const sortedValues = sortAssessmentValues(allValues);
    return sortedValues.map((value) => ({
      name: value,
      count: valueCounts[value] || 0,
    }));
  }, [distributionDataPoints]);

  const isLoading = isLoadingTimeSeries || isLoadingDistribution;
  const error = timeSeriesError || distributionError;
  const hasData = timeSeriesDataPoints.length > 0 || distributionDataPoints.length > 0;

  return {
    chartData,
    distributionChartData,
    isLoading,
    error,
    hasData,
  };
}
