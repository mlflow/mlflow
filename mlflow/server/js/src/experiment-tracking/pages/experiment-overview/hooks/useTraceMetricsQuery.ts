import { useQuery } from '../../../../common/utils/reactQueryHooks';
import { fetchOrFail, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import { catchNetworkErrorIfExists } from '../../../utils/NetworkUtils';
import {
  type QueryTraceMetricsRequest,
  type QueryTraceMetricsResponse,
  type MetricAggregation,
  MetricViewType,
} from '@databricks/web-shared/model-trace-explorer';

export const TRACE_METRICS_QUERY_KEY = 'traceMetrics';

/**
 * Query aggregated trace metrics for experiments
 */
async function queryTraceMetrics(params: QueryTraceMetricsRequest): Promise<QueryTraceMetricsResponse> {
  return fetchOrFail(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(params),
  })
    .then((res) => res.json())
    .catch(catchNetworkErrorIfExists);
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
  /** Optional: Filter expressions to apply (e.g. `trace.status="ERROR"`) */
  filters?: string[];
  /** Optional: Dimensions to group metrics by (e.g. `assessment_name`) */
  dimensions?: string[];
  /** Optional: Whether the query is enabled. Defaults to true. */
  enabled?: boolean;
}

export function useTraceMetricsQuery({
  experimentId,
  startTimeMs,
  endTimeMs,
  viewType,
  metricName,
  aggregations,
  timeIntervalSeconds,
  filters,
  dimensions,
  enabled = true,
}: UseTraceMetricsQueryParams) {
  const queryParams: QueryTraceMetricsRequest = {
    experiment_ids: [experimentId],
    view_type: viewType,
    metric_name: metricName,
    aggregations,
    time_interval_seconds: timeIntervalSeconds,
    start_time_ms: startTimeMs,
    end_time_ms: endTimeMs,
    filters,
    dimensions,
  };

  return useQuery({
    queryKey: [
      TRACE_METRICS_QUERY_KEY,
      experimentId,
      startTimeMs,
      endTimeMs,
      viewType,
      metricName,
      aggregations,
      timeIntervalSeconds,
      filters,
      dimensions,
    ],
    queryFn: async () => {
      const response = await queryTraceMetrics(queryParams);
      return response;
    },
    enabled: !!experimentId && enabled,
    refetchOnWindowFocus: false,
  });
}
