import { useMemo } from 'react';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  TraceDimensionKey,
  TraceStatus,
  SpanMetricKey,
} from '@databricks/web-shared/model-trace-explorer';
import type { QueryTraceMetricsResponse } from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { useOverviewChartContext } from '../OverviewChartContext';

export interface UserUsageData {
  /** User identifier (`mlflow.trace.user` metadata) */
  user: string;
  /** Total number of traces triggered by this user */
  traces: number;
  /** Number of distinct sessions */
  sessions: number;
  /** Sum of total tokens across the user's traces */
  totalTokens: number;
  /** Sum of span costs in USD */
  totalCost: number;
  /** Average trace latency in milliseconds */
  avgLatency: number;
  /** Error rate as a percentage (0-100) */
  errorRate: number;
}

export interface UseTopUsersDataResult {
  usersData: UserUsageData[];
  isLoading: boolean;
  error: unknown;
  hasData: boolean;
}

const USER_DIMENSION = TraceDimensionKey.TRACE_USER;

function valuesByUser(data: QueryTraceMetricsResponse | undefined, aggregation: AggregationType) {
  const result = new Map<string, number>();
  for (const dp of data?.data_points ?? []) {
    const user = dp.dimensions?.[USER_DIMENSION];
    const value = dp.values?.[aggregation];
    if (user && value !== undefined) {
      result.set(user, value);
    }
  }
  return result;
}

/**
 * Fetches per-user usage metrics (traces, sessions, tokens, cost, latency, error rate) for the
 * Usage tab, grouped by the `trace_user` dimension. Uses OverviewChartContext for the time range
 * and filters.
 */
export function useTopUsersData({ enabled = true }: { enabled?: boolean } = {}): UseTopUsersDataResult {
  const { experimentIds, startTimeMs, endTimeMs, filters } = useOverviewChartContext();
  const baseParams = { experimentIds, startTimeMs, endTimeMs, filters, enabled };

  const countQuery = useTraceMetricsQuery({
    ...baseParams,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    dimensions: [USER_DIMENSION, TraceDimensionKey.TRACE_STATUS],
  });

  const sessionsQuery = useTraceMetricsQuery({
    ...baseParams,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.SESSION_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    dimensions: [USER_DIMENSION],
  });

  const tokensQuery = useTraceMetricsQuery({
    ...baseParams,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TOTAL_TOKENS,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
    dimensions: [USER_DIMENSION],
  });

  const latencyQuery = useTraceMetricsQuery({
    ...baseParams,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.LATENCY,
    aggregations: [{ aggregation_type: AggregationType.AVG }],
    dimensions: [USER_DIMENSION],
  });

  const costQuery = useTraceMetricsQuery({
    ...baseParams,
    viewType: MetricViewType.SPANS,
    metricName: SpanMetricKey.TOTAL_COST,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
    dimensions: [USER_DIMENSION],
  });

  const usersData = useMemo(() => {
    const countsByUser = new Map<string, { total: number; errors: number }>();
    for (const dp of countQuery.data?.data_points ?? []) {
      const user = dp.dimensions?.[USER_DIMENSION];
      if (!user) continue;
      const count = dp.values?.[AggregationType.COUNT] || 0;
      const existing = countsByUser.get(user) || { total: 0, errors: 0 };
      existing.total += count;
      if (dp.dimensions?.[TraceDimensionKey.TRACE_STATUS] === TraceStatus.ERROR) {
        existing.errors += count;
      }
      countsByUser.set(user, existing);
    }

    const sessionsByUser = valuesByUser(sessionsQuery.data, AggregationType.COUNT);
    const tokensByUser = valuesByUser(tokensQuery.data, AggregationType.SUM);
    const latencyByUser = valuesByUser(latencyQuery.data, AggregationType.AVG);
    const costByUser = valuesByUser(costQuery.data, AggregationType.SUM);

    const result: UserUsageData[] = [];
    for (const [user, counts] of countsByUser.entries()) {
      result.push({
        user,
        traces: counts.total,
        sessions: sessionsByUser.get(user) || 0,
        totalTokens: tokensByUser.get(user) || 0,
        totalCost: costByUser.get(user) || 0,
        avgLatency: latencyByUser.get(user) || 0,
        errorRate: counts.total > 0 ? (counts.errors / counts.total) * 100 : 0,
      });
    }
    result.sort((a, b) => b.traces - a.traces);
    return result;
  }, [countQuery.data, sessionsQuery.data, tokensQuery.data, latencyQuery.data, costQuery.data]);

  const queries = [countQuery, sessionsQuery, tokensQuery, latencyQuery, costQuery];

  return {
    usersData,
    isLoading: queries.some((query) => query.isLoading),
    error: queries.find((query) => query.error)?.error,
    hasData: usersData.length > 0,
  };
}
