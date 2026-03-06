import { useMemo } from 'react';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  TIME_BUCKET_DIMENSION_KEY,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { TIME_UNIT_SECONDS, TimeUnit } from '../utils/timeUtils';

/**
 * Lightweight hook that determines the earliest trace timestamp for an experiment.
 * Queries trace count with yearly time buckets (very few results) and returns the
 * first bucket's timestamp. Used to compute an effective time range for "ALL" mode
 * so that time unit validation reflects the actual data span, not epoch-to-now.
 */
export function useEarliestTraceTimestamp(experimentIds: string[], enabled: boolean) {
  const { data } = useTraceMetricsQuery({
    experimentIds,
    startTimeMs: 0,
    endTimeMs: Date.now(),
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    timeIntervalSeconds: TIME_UNIT_SECONDS[TimeUnit.Year],
    enabled,
  });

  return useMemo(() => {
    const dataPoints = data?.data_points ?? [];
    if (dataPoints.length === 0) return undefined;

    // Data points are ordered by time bucket ascending — first one is the earliest
    const firstBucket = dataPoints[0]?.dimensions?.[TIME_BUCKET_DIMENSION_KEY];
    if (!firstBucket) return undefined;

    return new Date(firstBucket).getTime();
  }, [data?.data_points]);
}
