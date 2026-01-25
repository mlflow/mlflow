import { useMemo, useCallback } from 'react';
import { MetricViewType, AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import {
  formatTimestampForTraceMetrics,
  useTimestampValueMap,
  useChartZoom,
  ChartZoomState,
} from '../utils/chartUtils';
import { useOverviewChartContext } from '../OverviewChartContext';

export interface RequestsChartDataPoint {
  name: string;
  count: number;
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
  const { experimentId, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets, filters } = useOverviewChartContext();
  // Fetch trace count metrics grouped by time bucket
  const {
    data: traceCountData,
    isLoading,
    error,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    timeIntervalSeconds,
    filters,
  });

  const traceCountDataPoints = useMemo(() => traceCountData?.data_points || [], [traceCountData?.data_points]);

  // Get total requests
  const totalRequests = useMemo(
    () => traceCountDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.COUNT] || 0), 0),
    [traceCountDataPoints],
  );

  // Calculate average requests per time bucket
  const avgRequests = useMemo(() => totalRequests / timeBuckets.length, [totalRequests, timeBuckets.length]);

  // Create a map of counts by timestamp using shared utility
  const countExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => dp.values?.[AggregationType.COUNT] || 0,
    [],
  );
  const countByTimestamp = useTimestampValueMap(traceCountDataPoints, countExtractor);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => ({
      name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
      count: countByTimestamp.get(timestampMs) || 0,
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
