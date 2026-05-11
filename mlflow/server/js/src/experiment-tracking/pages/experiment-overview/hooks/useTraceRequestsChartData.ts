import { useMemo } from 'react';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  TraceDimensionKey,
  TIME_BUCKET_DIMENSION_KEY,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import type { ChartZoomState } from '../utils/chartUtils';
import { formatTimestampForTraceMetrics, useChartZoom } from '../utils/chartUtils';
import { useOverviewChartContext } from '../OverviewChartContext';

export interface RequestsChartDataPoint {
  name: string;
  count: number;
  /** Raw timestamp in milliseconds for navigation */
  timestampMs: number;
}

export interface UseTraceRequestsChartDataResult {
  /** Processed chart data with all time buckets filled */
  chartData: RequestsChartDataPoint[];
  /** Total number of requests in the time range */
  totalRequests: number;
  /** Average requests per time bucket (uses zoomed range if zoomed) */
  avgRequests: number;
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there are any data points */
  hasData: boolean;
  /** Zoom state and handlers */
  zoom: ChartZoomState<RequestsChartDataPoint>;
}

/**
 * Custom hook that fetches and processes requests chart data.
 * Encapsulates all data-fetching and processing logic for the requests chart.
 * Uses OverviewChartContext to get chart props.
 *
 * @returns Processed chart data, loading state, and error state
 */
export function useTraceRequestsChartData(): UseTraceRequestsChartDataResult {
  const { experimentIds, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets, filters } =
    useOverviewChartContext();
  // Fetch trace count metrics grouped by time bucket and trace status.
  // Adding TRACE_STATUS dimension allows React Query to deduplicate this query
  // with the errors chart, reducing the number of SQL queries.
  const {
    data: traceCountData,
    isLoading,
    error,
  } = useTraceMetricsQuery({
    experimentIds,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    timeIntervalSeconds,
    dimensions: [TraceDimensionKey.TRACE_STATUS],
    filters,
  });

  const traceCountDataPoints = useMemo(() => traceCountData?.data_points || [], [traceCountData?.data_points]);

  // Get total requests by summing across all status rows
  const totalRequests = useMemo(
    () => traceCountDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.COUNT] || 0), 0),
    [traceCountDataPoints],
  );

  // Calculate average requests per time bucket
  const avgRequests = useMemo(() => totalRequests / timeBuckets.length, [totalRequests, timeBuckets.length]);

  // Sum counts across all statuses per time bucket
  const countByTimestamp = useMemo(() => {
    const map = new Map<number, number>();
    for (const dp of traceCountDataPoints) {
      const ts = new Date(dp.dimensions?.[TIME_BUCKET_DIMENSION_KEY]).getTime();
      if (!isNaN(ts)) {
        map.set(ts, (map.get(ts) || 0) + (dp.values?.[AggregationType.COUNT] || 0));
      }
    }
    return map;
  }, [traceCountDataPoints]);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => ({
      name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
      count: countByTimestamp.get(timestampMs) || 0,
      timestampMs,
    }));
  }, [timeBuckets, countByTimestamp, timeIntervalSeconds]);

  // Zoom functionality
  const zoom = useChartZoom(chartData, 'name');

  // Calculate average for zoomed data when zoomed, otherwise use overall average
  const displayAvgRequests = useMemo(() => {
    if (!zoom.isZoomed) return avgRequests;
    if (zoom.zoomedData.length === 0) return 0;
    const total = zoom.zoomedData.reduce((sum, d) => sum + d.count, 0);
    return total / zoom.zoomedData.length;
  }, [zoom.isZoomed, zoom.zoomedData, avgRequests]);

  return {
    chartData,
    totalRequests,
    avgRequests: displayAvgRequests,
    isLoading,
    error,
    hasData: traceCountDataPoints.length > 0,
    zoom,
  };
}
