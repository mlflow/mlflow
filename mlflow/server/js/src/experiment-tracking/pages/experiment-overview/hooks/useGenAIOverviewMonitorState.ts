import { useMemo, useState } from 'react';
import {
  AggregationType,
  AssessmentDimensionKey,
  AssessmentFilterKey,
  AssessmentMetricKey,
  AssessmentTypeValue,
  MetricViewType,
  TIME_BUCKET_DIMENSION_KEY,
  createAssessmentFilter,
} from '@databricks/web-shared/model-trace-explorer';
import type { ScheduledScorer } from '../../experiment-scorers/types';
import { useGetScheduledScorers } from '../../experiment-scorers/hooks/useGetScheduledScorers';
import { useSqlWarehouseContextSafe } from '../../experiment-page-tabs/SqlWarehouseContext';
import { generateTimeBuckets } from '../utils/chartUtils';
import { GENAI_OVERVIEW_TIME_INTERVAL_SECONDS } from './useGenAIOverviewState';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';

const DAY_IN_MILLISECONDS = 24 * 60 * 60 * 1000;
const OVERVIEW_DAY_COUNT = 30;

const getPassFailDisplayValue = (value: unknown): 'pass' | 'fail' | undefined => {
  const normalizedValue = typeof value === 'string' ? value.trim().toLowerCase() : value;
  if (
    normalizedValue === true ||
    normalizedValue === 'true' ||
    normalizedValue === 'yes' ||
    normalizedValue === 'pass'
  ) {
    return 'pass';
  }
  if (
    normalizedValue === false ||
    normalizedValue === 'false' ||
    normalizedValue === 'no' ||
    normalizedValue === 'fail'
  ) {
    return 'fail';
  }
  return undefined;
};

const createOverviewTimeRange = () => {
  const now = new Date();
  const endTimeMs = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), 23, 59, 59, 999);
  return { startTimeMs: endTimeMs - OVERVIEW_DAY_COUNT * DAY_IN_MILLISECONDS + 1, endTimeMs };
};

export interface GenAIOverviewMonitorTrendPoint {
  timestampMs: number;
  value: number | null;
}

export type GenAIOverviewMonitorState =
  | { status: 'loading' }
  | { status: 'empty' }
  | {
      status: 'ready';
      onlineScorerCount: number;
      onlineScorerNames: string[];
      trendByScorerName?: Map<string, GenAIOverviewMonitorTrendPoint[]>;
      isTrendLoading?: boolean;
    }
  | { status: 'error' };

export const resolveGenAIOverviewMonitorState = ({
  scorers,
  isLoading,
  error,
}: {
  scorers: readonly ScheduledScorer[];
  isLoading: boolean;
  error: unknown;
}): GenAIOverviewMonitorState => {
  if (isLoading) return { status: 'loading' };
  if (error) return { status: 'error' };

  const onlineScorerNames = scorers
    .filter((scorer) => (scorer.sampleRate ?? 0) > 0)
    .map((scorer) => scorer.name)
    .sort((firstName, secondName) => firstName.localeCompare(secondName));
  return onlineScorerNames.length > 0
    ? { status: 'ready', onlineScorerCount: onlineScorerNames.length, onlineScorerNames }
    : { status: 'empty' };
};

export const useGenAIOverviewMonitorState = (experimentId: string): GenAIOverviewMonitorState => {
  const { data, isLoading, error } = useGetScheduledScorers(experimentId);
  const [timeRange] = useState(createOverviewTimeRange);
  const sqlWarehouseContext = useSqlWarehouseContextSafe();
  const warehouseRequired = Boolean(sqlWarehouseContext?.hasV4Location && !sqlWarehouseContext.warehouseId);
  const scorerState = useMemo(
    () =>
      resolveGenAIOverviewMonitorState({
        scorers: data?.scheduledScorers ?? [],
        isLoading,
        error,
      }),
    [data?.scheduledScorers, error, isLoading],
  );
  const onlineScorerNames = useMemo(
    () => (scorerState.status === 'ready' ? scorerState.onlineScorerNames : []),
    [scorerState],
  );
  const feedbackFilters = useMemo(
    () => [
      createAssessmentFilter(AssessmentFilterKey.TYPE, AssessmentTypeValue.FEEDBACK),
      `(${onlineScorerNames
        .map((scorerName) => createAssessmentFilter(AssessmentFilterKey.NAME, scorerName))
        .join(' OR ')})`,
    ],
    [onlineScorerNames],
  );
  const averageMetricsQuery = useTraceMetricsQuery({
    experimentIds: [experimentId],
    startTimeMs: timeRange.startTimeMs,
    endTimeMs: timeRange.endTimeMs,
    viewType: MetricViewType.ASSESSMENTS,
    metricName: AssessmentMetricKey.ASSESSMENT_VALUE,
    aggregations: [{ aggregation_type: AggregationType.AVG }],
    dimensions: [AssessmentDimensionKey.ASSESSMENT_NAME],
    timeIntervalSeconds: GENAI_OVERVIEW_TIME_INTERVAL_SECONDS,
    filters: feedbackFilters,
    enabled: onlineScorerNames.length > 0 && !warehouseRequired,
  });
  const passFailMetricsQuery = useTraceMetricsQuery({
    experimentIds: [experimentId],
    startTimeMs: timeRange.startTimeMs,
    endTimeMs: timeRange.endTimeMs,
    viewType: MetricViewType.ASSESSMENTS,
    metricName: AssessmentMetricKey.ASSESSMENT_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    dimensions: [AssessmentDimensionKey.ASSESSMENT_NAME, AssessmentDimensionKey.ASSESSMENT_VALUE],
    timeIntervalSeconds: GENAI_OVERVIEW_TIME_INTERVAL_SECONDS,
    filters: feedbackFilters,
    enabled: onlineScorerNames.length > 0 && !warehouseRequired,
  });

  const trendByScorerName = useMemo(() => {
    const valuesByNameAndTime = new Map<string, Map<number, number | null>>();
    for (const dataPoint of averageMetricsQuery.data?.data_points ?? []) {
      const scorerName = dataPoint.dimensions?.[AssessmentDimensionKey.ASSESSMENT_NAME];
      const timeBucket = dataPoint.dimensions?.[TIME_BUCKET_DIMENSION_KEY];
      if (!scorerName || !timeBucket) continue;
      const timestampMs = new Date(timeBucket).getTime();
      if (Number.isNaN(timestampMs)) continue;
      const valuesByTime = valuesByNameAndTime.get(scorerName) ?? new Map<number, number | null>();
      valuesByTime.set(timestampMs, dataPoint.values?.[AggregationType.AVG] ?? null);
      valuesByNameAndTime.set(scorerName, valuesByTime);
    }

    const passFailCountsByNameAndTime = new Map<string, Map<number, { passCount: number; failCount: number }>>();
    for (const dataPoint of passFailMetricsQuery.data?.data_points ?? []) {
      const scorerName = dataPoint.dimensions?.[AssessmentDimensionKey.ASSESSMENT_NAME];
      const assessmentValue = dataPoint.dimensions?.[AssessmentDimensionKey.ASSESSMENT_VALUE];
      const timeBucket = dataPoint.dimensions?.[TIME_BUCKET_DIMENSION_KEY];
      const count = dataPoint.values?.[AggregationType.COUNT];
      const outcome = getPassFailDisplayValue(assessmentValue);
      if (!scorerName || !timeBucket || !outcome || typeof count !== 'number' || !Number.isFinite(count)) continue;
      const timestampMs = new Date(timeBucket).getTime();
      if (Number.isNaN(timestampMs)) continue;
      const countsByTime = passFailCountsByNameAndTime.get(scorerName) ?? new Map();
      const counts = countsByTime.get(timestampMs) ?? { passCount: 0, failCount: 0 };
      counts[outcome === 'pass' ? 'passCount' : 'failCount'] += count;
      countsByTime.set(timestampMs, counts);
      passFailCountsByNameAndTime.set(scorerName, countsByTime);
    }

    const timeBuckets = generateTimeBuckets(
      timeRange.startTimeMs,
      timeRange.endTimeMs,
      GENAI_OVERVIEW_TIME_INTERVAL_SECONDS,
    );
    return new Map(
      onlineScorerNames.map((scorerName) => {
        const valuesByTime = valuesByNameAndTime.get(scorerName);
        const passFailCountsByTime = passFailCountsByNameAndTime.get(scorerName);
        return [
          scorerName,
          timeBuckets.map((timestampMs) => {
            const counts = passFailCountsByTime?.get(timestampMs);
            const totalCount = (counts?.passCount ?? 0) + (counts?.failCount ?? 0);
            return {
              timestampMs,
              value: totalCount > 0 ? (counts?.passCount ?? 0) / totalCount : (valuesByTime?.get(timestampMs) ?? null),
            };
          }),
        ];
      }),
    );
  }, [
    averageMetricsQuery.data?.data_points,
    onlineScorerNames,
    passFailMetricsQuery.data?.data_points,
    timeRange.endTimeMs,
    timeRange.startTimeMs,
  ]);

  if (scorerState.status !== 'ready') return scorerState;
  return {
    ...scorerState,
    trendByScorerName,
    isTrendLoading:
      averageMetricsQuery.isLoading ||
      averageMetricsQuery.isFetching ||
      passFailMetricsQuery.isLoading ||
      passFailMetricsQuery.isFetching,
  };
};
