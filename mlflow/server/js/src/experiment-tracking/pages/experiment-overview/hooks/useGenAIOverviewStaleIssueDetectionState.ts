import { useMemo } from 'react';
import {
  AggregationType,
  MetricViewType,
  TraceMetricKey,
  type MetricDataPoint,
} from '@databricks/web-shared/model-trace-explorer';
import { useMonitoringConfig } from '../../../hooks/useMonitoringConfig';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';

const DAY_IN_MILLISECONDS = 24 * 60 * 60 * 1000;
const EMPTY_TRACE_COUNT_REFETCH_INTERVAL_MS = 60 * 1000;
export const GENAI_OVERVIEW_STALE_ISSUE_DETECTION_DAY_COUNT = 7;

export type GenAIOverviewStaleIssueDetectionState =
  | { status: 'hidden' }
  | { status: 'loading' }
  | {
      status: 'ready';
      latestRunUuid: string;
      elapsedDays: number;
      newTraceCount: number;
    };

export interface ResolveGenAIOverviewStaleIssueDetectionStateParams {
  enabled: boolean;
  latestRunUuid?: string;
  latestRunCompletedAtMs?: number;
  nowMs: number;
  dataPoints?: MetricDataPoint[];
  isLoading: boolean;
  error: unknown;
}

export const resolveGenAIOverviewStaleIssueDetectionState = ({
  enabled,
  latestRunUuid,
  latestRunCompletedAtMs,
  nowMs,
  dataPoints,
  isLoading,
  error,
}: ResolveGenAIOverviewStaleIssueDetectionStateParams): GenAIOverviewStaleIssueDetectionState => {
  if (!enabled || !latestRunUuid || latestRunCompletedAtMs === undefined) return { status: 'hidden' };
  const elapsedDays = Math.floor(Math.max(0, nowMs - latestRunCompletedAtMs) / DAY_IN_MILLISECONDS);
  if (elapsedDays < GENAI_OVERVIEW_STALE_ISSUE_DETECTION_DAY_COUNT) return { status: 'hidden' };
  if (isLoading) return { status: 'loading' };
  if (error || !dataPoints) return { status: 'hidden' };
  const newTraceCount = dataPoints.reduce(
    (total, dataPoint) => total + (dataPoint.values?.[AggregationType.COUNT] ?? 0),
    0,
  );
  return newTraceCount > 0 ? { status: 'ready', latestRunUuid, elapsedDays, newTraceCount } : { status: 'hidden' };
};

export const useGenAIOverviewStaleIssueDetectionState = ({
  experimentId,
  latestRunUuid,
  latestRunCompletedAtMs,
  enabled,
}: {
  experimentId: string;
  latestRunUuid?: string;
  latestRunCompletedAtMs?: number;
  enabled: boolean;
}) => {
  const { dateNow } = useMonitoringConfig();
  const nowMs = dateNow.getTime();
  const elapsedDays =
    latestRunCompletedAtMs === undefined
      ? 0
      : Math.floor(Math.max(0, nowMs - latestRunCompletedAtMs) / DAY_IN_MILLISECONDS);
  const shouldQueryTraceCount =
    enabled &&
    Boolean(experimentId && latestRunUuid && latestRunCompletedAtMs !== undefined) &&
    elapsedDays >= GENAI_OVERVIEW_STALE_ISSUE_DETECTION_DAY_COUNT;
  const { data, isLoading, isFetching, error } = useTraceMetricsQuery({
    experimentIds: [experimentId],
    startTimeMs: latestRunCompletedAtMs === undefined ? undefined : latestRunCompletedAtMs + 1,
    endTimeMs: nowMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    enabled: shouldQueryTraceCount,
    refetchIntervalWhileEmptyMs: EMPTY_TRACE_COUNT_REFETCH_INTERVAL_MS,
  });

  return useMemo(
    () =>
      resolveGenAIOverviewStaleIssueDetectionState({
        enabled,
        latestRunUuid,
        latestRunCompletedAtMs,
        nowMs,
        dataPoints: data?.data_points,
        isLoading: isLoading || isFetching,
        error,
      }),
    [data?.data_points, enabled, error, isFetching, isLoading, latestRunCompletedAtMs, latestRunUuid, nowMs],
  );
};
