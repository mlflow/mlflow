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

/** Requests no built-in MLflow feature made, such as calls from applications or the playground. */
export const OTHER_CALLERS = 'other';

export type GatewayFeatureUsageKey = GatewayFeature | typeof OTHER_CALLERS;

// MLflow features in their fixed order, then everything else.
const USAGE_ORDER: GatewayFeatureUsageKey[] = [...GATEWAY_FEATURES, OTHER_CALLERS];

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
): Map<GatewayFeatureUsageKey, number> => {
  const totals = new Map<GatewayFeatureUsageKey, number>();
  for (const dataPoint of dataPoints ?? []) {
    const caller = dataPoint.dimensions?.[dimensionKey];
    if (!caller) continue;
    const feature = isGatewayFeature(caller) ? caller : OTHER_CALLERS;
    totals.set(feature, (totals.get(feature) ?? 0) + (dataPoint.values?.[aggregation] ?? 0));
  }
  return totals;
};

/**
 * Gateway requests, tokens and cost broken down by the built-in MLflow feature that made them.
 * Callers that are not MLflow features are combined into a single `OTHER_CALLERS` entry.
 */
export function useGatewayFeatureUsageData() {
  const { experimentIds, startTimeMs, endTimeMs, filters } = useOverviewChartContext();
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

  const usage = useMemo((): GatewayFeatureUsage[] => {
    const requests = sumByFeature(
      requestsQuery.data?.data_points,
      TraceDimensionKey.GATEWAY_CALLER,
      AggregationType.COUNT,
    );
    const tokens = sumByFeature(tokensQuery.data?.data_points, TraceDimensionKey.GATEWAY_CALLER, AggregationType.SUM);
    const cost = sumByFeature(costQuery.data?.data_points, SpanDimensionKey.GATEWAY_CALLER, AggregationType.SUM);

    return USAGE_ORDER.filter((feature) => requests.has(feature)).map((feature) => ({
      feature,
      requests: requests.get(feature) ?? 0,
      tokens: tokens.get(feature) ?? 0,
      cost: cost.get(feature) ?? 0,
    }));
  }, [requestsQuery.data?.data_points, tokensQuery.data?.data_points, costQuery.data?.data_points]);

  return {
    usage,
    hasFeatureUsage: usage.some(({ feature }) => feature !== OTHER_CALLERS),
    isLoading: requestsQuery.isLoading || tokensQuery.isLoading || costQuery.isLoading,
    error: requestsQuery.error || tokensQuery.error || costQuery.error,
  };
}
