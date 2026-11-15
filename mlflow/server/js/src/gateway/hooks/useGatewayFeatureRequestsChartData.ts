import { useMemo } from 'react';
import {
  AggregationType,
  MetricViewType,
  TIME_BUCKET_DIMENSION_KEY,
  TraceDimensionKey,
  TraceMetricKey,
  type MetricDataPoint,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from '../../experiment-tracking/pages/experiment-overview/hooks/useTraceMetricsQuery';
import { useOverviewChartContext } from '../../experiment-tracking/pages/experiment-overview/OverviewChartContext';
import { formatTimestampForTraceMetrics } from '../../experiment-tracking/pages/experiment-overview/utils/chartUtils';
import {
  GATEWAY_FEATURES,
  OTHER_CALLERS,
  isGatewayFeature,
  type GatewayFeature,
  type GatewayFeatureUsageKey,
} from '../utils/gatewayFeatures';

export interface GatewayFeatureRequestsDataPoint {
  timestamp: string;
  timestampMs: number;
  values: Record<GatewayFeatureUsageKey, number>;
}

const SERIES_ORDER: GatewayFeatureUsageKey[] = [...GATEWAY_FEATURES, OTHER_CALLERS];

const bucketTimestampMs = (dataPoint: MetricDataPoint): number =>
  new Date(dataPoint.dimensions?.[TIME_BUCKET_DIMENSION_KEY] ?? '').getTime();

const requestCount = (dataPoint: MetricDataPoint): number => dataPoint.values?.[AggregationType.COUNT] ?? 0;

/**
 * Gateway requests over time, broken down by the built-in MLflow feature that made them. Like the
 * Usage by Feature table, "Other" is the rest of each time bucket's requests, so it includes traces
 * that have no caller and the bars add up to the Requests chart.
 */
export function useGatewayFeatureRequestsChartData() {
  const { experimentIds, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets, filters } =
    useOverviewChartContext();
  const queryScope = {
    experimentIds,
    startTimeMs,
    endTimeMs,
    filters,
    timeIntervalSeconds,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
  };

  const byCallerQuery = useTraceMetricsQuery({ ...queryScope, dimensions: [TraceDimensionKey.GATEWAY_CALLER] });
  // The same query as the Requests chart, so React Query shares its result.
  const totalsQuery = useTraceMetricsQuery({ ...queryScope, dimensions: [TraceDimensionKey.TRACE_STATUS] });

  const { chartData, series } = useMemo(() => {
    const featureCountsByBucket = new Map<number, Map<GatewayFeature, number>>();
    for (const dataPoint of byCallerQuery.data?.data_points ?? []) {
      const caller = dataPoint.dimensions?.[TraceDimensionKey.GATEWAY_CALLER];
      if (!caller || !isGatewayFeature(caller)) continue;
      const timestampMs = bucketTimestampMs(dataPoint);
      const counts = featureCountsByBucket.get(timestampMs) ?? new Map<GatewayFeature, number>();
      counts.set(caller, (counts.get(caller) ?? 0) + requestCount(dataPoint));
      featureCountsByBucket.set(timestampMs, counts);
    }

    const totalsByBucket = new Map<number, number>();
    for (const dataPoint of totalsQuery.data?.data_points ?? []) {
      const timestampMs = bucketTimestampMs(dataPoint);
      totalsByBucket.set(timestampMs, (totalsByBucket.get(timestampMs) ?? 0) + requestCount(dataPoint));
    }

    const points = timeBuckets.map((timestampMs): GatewayFeatureRequestsDataPoint => {
      const counts = featureCountsByBucket.get(timestampMs);
      const values = { [OTHER_CALLERS]: totalsByBucket.get(timestampMs) ?? 0 } as Record<
        GatewayFeatureUsageKey,
        number
      >;
      for (const feature of GATEWAY_FEATURES) {
        values[feature] = counts?.get(feature) ?? 0;
        values[OTHER_CALLERS] -= values[feature];
      }
      values[OTHER_CALLERS] = Math.max(0, values[OTHER_CALLERS]);
      return { timestamp: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds), timestampMs, values };
    });

    return {
      chartData: points,
      series: SERIES_ORDER.filter((key) => points.some(({ values }) => values[key] > 0)),
    };
  }, [byCallerQuery.data?.data_points, totalsQuery.data?.data_points, timeBuckets, timeIntervalSeconds]);

  return {
    chartData,
    series,
    hasData: series.length > 0,
    isLoading: byCallerQuery.isLoading || totalsQuery.isLoading,
    error: byCallerQuery.error || totalsQuery.error,
  };
}
