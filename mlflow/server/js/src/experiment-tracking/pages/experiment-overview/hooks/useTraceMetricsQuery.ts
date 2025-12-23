import { useQuery } from '../../../../common/utils/reactQueryHooks';
import { MlflowService } from '../../../sdk/MlflowService';
import {
  type QueryTraceMetricsRequest,
  type MetricAggregation,
  MetricViewType,
} from '@databricks/web-shared/model-trace-explorer';

export const TRACE_METRICS_QUERY_KEY = 'traceMetrics';

// Time intervals in seconds
const MINUTE_IN_SECONDS = 60;
const HOUR_IN_SECONDS = 3600;
const DAY_IN_SECONDS = 86400;
const MONTH_IN_SECONDS = 2592000;

/**
 * Calculate the appropriate time interval based on the time range.
 * Returns day-level interval if times are not provided.
 * - <= 1 hour: minute level
 * - <= 24 hours: hour level
 * - <= 1 month: day level
 * - > 1 month: month level
 */
export function calculateTimeInterval(startTimeMs?: number, endTimeMs?: number): number {
  if (!startTimeMs || !endTimeMs) {
    return DAY_IN_SECONDS;
  }

  const durationMs = endTimeMs - startTimeMs;
  const durationSeconds = durationMs / 1000;

  if (durationSeconds <= HOUR_IN_SECONDS) {
    return MINUTE_IN_SECONDS;
  } else if (durationSeconds <= DAY_IN_SECONDS) {
    return HOUR_IN_SECONDS;
  } else if (durationSeconds <= MONTH_IN_SECONDS) {
    return DAY_IN_SECONDS;
  } else {
    return MONTH_IN_SECONDS;
  }
}

interface UseTraceMetricsQueryParams {
  experimentId: string;
  startTimeMs?: number;
  endTimeMs?: number;
  viewType: MetricViewType;
  metricName: string;
  aggregations: MetricAggregation[];
  /** Optional: Time interval for grouping. If not provided, no time grouping is applied. */
  timeIntervalSeconds?: number;
}

export function useTraceMetricsQuery({
  experimentId,
  startTimeMs,
  endTimeMs,
  viewType,
  metricName,
  aggregations,
  timeIntervalSeconds,
}: UseTraceMetricsQueryParams) {
  const queryParams: QueryTraceMetricsRequest = {
    experiment_ids: [experimentId],
    view_type: viewType,
    metric_name: metricName,
    aggregations,
    time_interval_seconds: timeIntervalSeconds,
    start_time_ms: startTimeMs,
    end_time_ms: endTimeMs,
  };

  return useQuery({
    queryKey: [
      TRACE_METRICS_QUERY_KEY,
      experimentId,
      startTimeMs,
      endTimeMs,
      viewType,
      metricName,
      JSON.stringify(aggregations),
      timeIntervalSeconds,
    ],
    queryFn: async () => {
      const response = await MlflowService.queryTraceMetrics(queryParams);
      return response;
    },
    enabled: !!experimentId,
    refetchOnWindowFocus: false,
  });
}
