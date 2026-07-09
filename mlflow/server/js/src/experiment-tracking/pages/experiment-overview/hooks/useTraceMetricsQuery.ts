import { useQuery } from '../../../../common/utils/reactQueryHooks';
import { fetchOrFail, getAjaxUrl } from '../../../../common/utils/FetchUtils';
import { catchNetworkErrorIfExists } from '../../../utils/NetworkUtils';
import type { MetricViewType } from '@databricks/web-shared/model-trace-explorer';
import {
  type QueryTraceMetricsRequest,
  type QueryTraceMetricsResponse,
  type MetricAggregation,
} from '@databricks/web-shared/model-trace-explorer';
import { shouldUseTracesV4API } from '@databricks/web-shared/genai-traces-table';
import { shouldEnableBatchedTokenMetricQueries } from '../../../../common/utils/FeatureUtils';
import { useSqlWarehouseContextSafe } from '../../experiment-page-tabs/SqlWarehouseContext';

const TRACE_METRICS_QUERY_KEY = 'traceMetrics';

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

/**
 * Query aggregated trace metrics via the V4 Databricks API.
 */
async function queryTraceMetricsV4(
  params: QueryTraceMetricsRequest,
  sqlWarehouseId?: string | null,
): Promise<QueryTraceMetricsResponse> {
  const { experiment_ids, ...rest } = params;
  const v4Payload = {
    ...rest,
    locations: experiment_ids.map((id) => ({
      type: 'MLFLOW_EXPERIMENT',
      mlflow_experiment: { experiment_id: id },
    })),
    ...(sqlWarehouseId ? { sql_warehouse_id: sqlWarehouseId } : {}),
  };
  return fetchOrFail(getAjaxUrl('ajax-api/4.0/mlflow/traces/metrics'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(v4Payload),
  })
    .then((res) => res.json())
    .catch(catchNetworkErrorIfExists);
}

interface UseTraceMetricsQueryParams {
  experimentIds: string[];
  startTimeMs?: number;
  endTimeMs?: number;
  viewType: MetricViewType;
  /** @deprecated Use metricNames instead. */
  metricName?: string;
  /** The name(s) of the metric(s) to query. Replaces metricName. */
  metricNames?: string[];
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
  experimentIds,
  startTimeMs,
  endTimeMs,
  viewType,
  metricName,
  metricNames,
  aggregations,
  timeIntervalSeconds,
  filters,
  dimensions,
  enabled = true,
}: UseTraceMetricsQueryParams) {
  const useV4 = shouldUseTracesV4API();
  const sqlWarehouseContext = useSqlWarehouseContextSafe();
  const sqlWarehouseId = sqlWarehouseContext?.warehouseId;

  // When batching is enabled, auto-promote metricName (singular) to metricNames (plural)
  // so the backend always receives metric_names, even for single-metric queries.
  const isBatchingEnabled = Boolean(shouldEnableBatchedTokenMetricQueries());
  const resolvedMetricNames = metricNames ?? (isBatchingEnabled && metricName ? [metricName] : undefined);
  const resolvedMetricName = resolvedMetricNames ? undefined : metricName;

  const hasMetric = !!resolvedMetricNames?.length || !!resolvedMetricName;

  const queryParams: QueryTraceMetricsRequest = {
    experiment_ids: experimentIds,
    view_type: viewType,
    metric_name: resolvedMetricName,
    metric_names: resolvedMetricNames,
    aggregations,
    time_interval_seconds: timeIntervalSeconds,
    start_time_ms: startTimeMs,
    end_time_ms: endTimeMs,
    filters,
    dimensions,
  };

  // V4 backend requires start_time_ms, end_time_ms, and sql_warehouse_id; disable queries that omit them.
  const queryEnabled =
    experimentIds.length > 0 &&
    hasMetric &&
    enabled &&
    (!useV4 || (startTimeMs !== undefined && endTimeMs !== undefined && !!sqlWarehouseId));

  const result = useQuery({
    queryKey: [
      TRACE_METRICS_QUERY_KEY,
      experimentIds,
      startTimeMs,
      endTimeMs,
      viewType,
      metricName,
      metricNames,
      aggregations,
      timeIntervalSeconds,
      filters,
      dimensions,
      sqlWarehouseId,
    ],
    queryFn: async () => {
      if (useV4) {
        return queryTraceMetricsV4(queryParams, sqlWarehouseId);
      }
      return queryTraceMetrics(queryParams);
    },
    enabled: queryEnabled,
    refetchOnWindowFocus: false,
  });

  return { ...result, isLoading: result.isLoading && queryEnabled };
}
