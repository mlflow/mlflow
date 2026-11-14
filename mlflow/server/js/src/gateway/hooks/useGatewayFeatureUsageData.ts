import { useMemo } from 'react';
import {
  AggregationType,
  MetricViewType,
  SpanDimensionKey,
  SpanMetricKey,
  TraceDimensionKey,
  TraceMetricKey,
  type MetricDataPoint,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from '../../experiment-tracking/pages/experiment-overview/hooks/useTraceMetricsQuery';
import { useOverviewChartContext } from '../../experiment-tracking/pages/experiment-overview/OverviewChartContext';
import { GATEWAY_FEATURES, isGatewayFeature, type GatewayFeature } from '../utils/gatewayFeatures';

/**
 * Requests no built-in MLflow feature made, such as calls from applications or the playground, and
 * requests recorded without a caller.
 */
export const OTHER_CALLERS = 'other';

export type GatewayFeatureUsageKey = GatewayFeature | typeof OTHER_CALLERS;

export interface GatewayFeatureUsage {
  feature: GatewayFeatureUsageKey;
  requests: number;
  tokens: number;
  cost: number;
}

const sumByFeature = (
  dataPoints: MetricDataPoint[] | undefined,
  dimensionKey: string,
  aggregation: AggregationType,
): Map<GatewayFeature, number> => {
  const totals = new Map<GatewayFeature, number>();
  for (const dataPoint of dataPoints ?? []) {
    const caller = dataPoint.dimensions?.[dimensionKey];
    if (!caller || !isGatewayFeature(caller)) continue;
    totals.set(caller, (totals.get(caller) ?? 0) + (dataPoint.values?.[aggregation] ?? 0));
  }
  return totals;
};

const sumValues = (dataPoints: MetricDataPoint[] | undefined, aggregation: AggregationType): number =>
  (dataPoints ?? []).reduce((sum, dataPoint) => sum + (dataPoint.values?.[aggregation] ?? 0), 0);

/**
 * Gateway requests, tokens and cost broken down by the built-in MLflow feature that made them.
 * The `OTHER_CALLERS` entry is the rest of the totals rather than a sum of the other callers,
 * because grouping by caller leaves out traces that have none, and the rows should add up to the
 * totals shown by the other charts.
 */
export function useGatewayFeatureUsageData() {
  const { experimentIds, startTimeMs, endTimeMs, timeIntervalSeconds, filters } = useOverviewChartContext();
  const queryScope = { experimentIds, startTimeMs, endTimeMs, filters };

  const requestsQuery = useTraceMetricsQuery({
    ...queryScope,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    dimensions: [TraceDimensionKey.GATEWAY_CALLER],
  });
  const tokensQuery = useTraceMetricsQuery({
    ...queryScope,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TOTAL_TOKENS,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
    dimensions: [TraceDimensionKey.GATEWAY_CALLER],
  });
  const costQuery = useTraceMetricsQuery({
    ...queryScope,
    viewType: MetricViewType.SPANS,
    metricName: SpanMetricKey.TOTAL_COST,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
    dimensions: [SpanDimensionKey.GATEWAY_CALLER],
  });

  // The request and token totals use the same queries as the Requests and Token Usage charts, so
  // React Query shares their results.
  const totalRequestsQuery = useTraceMetricsQuery({
    ...queryScope,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    timeIntervalSeconds,
    dimensions: [TraceDimensionKey.TRACE_STATUS],
  });
  const totalTokensQuery = useTraceMetricsQuery({
    ...queryScope,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TOTAL_TOKENS,
    aggregations: [{ aggregation_type: AggregationType.SUM }, { aggregation_type: AggregationType.AVG }],
  });
  const totalCostQuery = useTraceMetricsQuery({
    ...queryScope,
    viewType: MetricViewType.SPANS,
    metricName: SpanMetricKey.TOTAL_COST,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
  });

  const queries = [requestsQuery, tokensQuery, costQuery, totalRequestsQuery, totalTokensQuery, totalCostQuery];

  const usage = useMemo((): GatewayFeatureUsage[] => {
    const requests = sumByFeature(
      requestsQuery.data?.data_points,
      TraceDimensionKey.GATEWAY_CALLER,
      AggregationType.COUNT,
    );
    const tokens = sumByFeature(tokensQuery.data?.data_points, TraceDimensionKey.GATEWAY_CALLER, AggregationType.SUM);
    const cost = sumByFeature(costQuery.data?.data_points, SpanDimensionKey.GATEWAY_CALLER, AggregationType.SUM);

    const featureUsage: GatewayFeatureUsage[] = GATEWAY_FEATURES.filter((feature) => requests.has(feature)).map(
      (feature) => ({
        feature,
        requests: requests.get(feature) ?? 0,
        tokens: tokens.get(feature) ?? 0,
        cost: cost.get(feature) ?? 0,
      }),
    );
    // Subtracting floating-point sums can leave a tiny negative remainder, so clamp at zero.
    const remainder = (total: number, column: 'requests' | 'tokens' | 'cost') =>
      Math.max(
        0,
        featureUsage.reduce((rest, row) => rest - row[column], total),
      );

    const otherRequests = remainder(sumValues(totalRequestsQuery.data?.data_points, AggregationType.COUNT), 'requests');
    if (otherRequests === 0) {
      return featureUsage;
    }
    return [
      ...featureUsage,
      {
        feature: OTHER_CALLERS,
        requests: otherRequests,
        tokens: remainder(sumValues(totalTokensQuery.data?.data_points, AggregationType.SUM), 'tokens'),
        cost: remainder(sumValues(totalCostQuery.data?.data_points, AggregationType.SUM), 'cost'),
      },
    ];
  }, [
    requestsQuery.data?.data_points,
    tokensQuery.data?.data_points,
    costQuery.data?.data_points,
    totalRequestsQuery.data?.data_points,
    totalTokensQuery.data?.data_points,
    totalCostQuery.data?.data_points,
  ]);

  return {
    usage,
    hasFeatureUsage: usage.some(({ feature }) => feature !== OTHER_CALLERS),
    isLoading: queries.some((query) => query.isLoading),
    error: queries.find((query) => query.error)?.error,
  };
}
