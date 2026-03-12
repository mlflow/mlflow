import { useMemo } from 'react';
import {
  MetricViewType,
  AggregationType,
  SpanMetricKey,
  SpanDimensionKey,
  TIME_BUCKET_DIMENSION_KEY,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { formatTimestampForTraceMetrics } from '../utils/chartUtils';
import { useOverviewChartContext } from '../OverviewChartContext';
import type { CostDimension } from './useTraceCostDimension';

export interface CostOverTimeDataPoint {
  name: string;
  [key: string]: number | string;
}

export interface UseTraceCostOverTimeChartDataResult {
  /** Processed chart data with all time buckets filled */
  chartData: CostOverTimeDataPoint[];
  /** List of unique dimension values for creating chart lines */
  dimensionValues: string[];
  /** Total cost across all dimension values in the time range */
  totalCost: number;
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there are any data points */
  hasData: boolean;
}

/**
 * Custom hook that fetches and processes cost over time chart data grouped by a dimension.
 * Encapsulates all data-fetching and processing logic for the cost over time chart.
 *
 * @returns Processed chart data, loading state, and error state
 */
export function useTraceCostOverTimeChartData(dimension: CostDimension = 'model'): UseTraceCostOverTimeChartDataResult {
  const { experimentIds, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets, filters } =
    useOverviewChartContext();

  const dimensionKey = dimension === 'model' ? SpanDimensionKey.MODEL_NAME : SpanDimensionKey.MODEL_PROVIDER;

  // Fetch total cost grouped by dimension and time
  const { data, isLoading, error } = useTraceMetricsQuery({
    experimentIds,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.SPANS,
    metricName: SpanMetricKey.TOTAL_COST,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
    dimensions: [dimensionKey],
    timeIntervalSeconds,
    filters,
  });

  const dataPoints = useMemo(() => data?.data_points || [], [data?.data_points]);

  // Extract unique dimension values from the data
  const dimensionValues = useMemo(() => {
    const names = new Set<string>();
    for (const dp of dataPoints) {
      const value = dp.dimensions?.[dimensionKey];
      if (value) {
        names.add(value);
      }
    }
    return Array.from(names).sort();
  }, [dataPoints, dimensionKey]);

  // Create a map of timestamp -> dimensionValue -> cost
  const costByTimestampAndDimension = useMemo(() => {
    const map = new Map<number, Map<string, number>>();

    for (const dp of dataPoints) {
      const timeBucket = dp.dimensions?.[TIME_BUCKET_DIMENSION_KEY];
      const dimValue = dp.dimensions?.[dimensionKey];
      const cost = dp.values?.[AggregationType.SUM] || 0;

      if (timeBucket && dimValue) {
        const timestampMs = new Date(timeBucket).getTime();
        if (!map.has(timestampMs)) {
          map.set(timestampMs, new Map());
        }
        map.get(timestampMs)!.set(dimValue, cost);
      }
    }

    return map;
  }, [dataPoints, dimensionKey]);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => {
      const costs = costByTimestampAndDimension.get(timestampMs);
      const dataPoint: CostOverTimeDataPoint = {
        name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
      };

      for (const dimValue of dimensionValues) {
        dataPoint[dimValue] = costs?.get(dimValue) || 0;
      }

      return dataPoint;
    });
  }, [timeBuckets, costByTimestampAndDimension, dimensionValues, timeIntervalSeconds]);

  // Calculate total cost across all dimension values
  const totalCost = useMemo(() => {
    return dataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.SUM] || 0), 0);
  }, [dataPoints]);

  return {
    chartData,
    dimensionValues,
    totalCost,
    isLoading,
    error,
    hasData: dataPoints.length > 0,
  };
}
