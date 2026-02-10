import { useMemo, useCallback } from 'react';
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

export interface CostOverTimeDataPoint {
  name: string;
  [modelName: string]: number | string;
}

export interface UseTraceCostOverTimeChartDataResult {
  /** Processed chart data with all time buckets filled */
  chartData: CostOverTimeDataPoint[];
  /** List of unique model names for creating chart lines */
  modelNames: string[];
  /** Total cost across all models in the time range */
  totalCost: number;
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there are any data points */
  hasData: boolean;
}

/**
 * Custom hook that fetches and processes cost over time chart data grouped by model name.
 * Encapsulates all data-fetching and processing logic for the cost over time chart.
 *
 * @returns Processed chart data, loading state, and error state
 */
export function useTraceCostOverTimeChartData(): UseTraceCostOverTimeChartDataResult {
  const { experimentId, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets } = useOverviewChartContext();

  // Fetch total cost grouped by model name and time
  const { data, isLoading, error } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.SPANS,
    metricName: SpanMetricKey.TOTAL_COST,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
    dimensions: [SpanDimensionKey.MODEL_NAME],
    timeIntervalSeconds,
  });

  const dataPoints = useMemo(() => data?.data_points || [], [data?.data_points]);

  // Extract unique model names from the data
  const modelNames = useMemo(() => {
    const names = new Set<string>();
    for (const dp of dataPoints) {
      const modelName = dp.dimensions?.[SpanDimensionKey.MODEL_NAME];
      if (modelName) {
        names.add(modelName);
      }
    }
    // Sort model names for consistent ordering
    return Array.from(names).sort();
  }, [dataPoints]);

  // Create a map of timestamp -> modelName -> cost
  const costByTimestampAndModel = useMemo(() => {
    const map = new Map<number, Map<string, number>>();

    for (const dp of dataPoints) {
      const timeBucket = dp.dimensions?.[TIME_BUCKET_DIMENSION_KEY];
      const modelName = dp.dimensions?.[SpanDimensionKey.MODEL_NAME];
      const cost = dp.values?.[AggregationType.SUM] || 0;

      if (timeBucket && modelName) {
        const timestampMs = new Date(timeBucket).getTime();
        if (!map.has(timestampMs)) {
          map.set(timestampMs, new Map());
        }
        map.get(timestampMs)!.set(modelName, cost);
      }
    }

    return map;
  }, [dataPoints]);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => {
      const modelCosts = costByTimestampAndModel.get(timestampMs);
      const dataPoint: CostOverTimeDataPoint = {
        name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
      };

      // Add cost for each model (0 if no data for that time bucket)
      for (const modelName of modelNames) {
        dataPoint[modelName] = modelCosts?.get(modelName) || 0;
      }

      return dataPoint;
    });
  }, [timeBuckets, costByTimestampAndModel, modelNames, timeIntervalSeconds]);

  // Calculate total cost across all models
  const totalCost = useMemo(() => {
    return dataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.SUM] || 0), 0);
  }, [dataPoints]);

  return {
    chartData,
    modelNames,
    totalCost,
    isLoading,
    error,
    hasData: dataPoints.length > 0,
  };
}
