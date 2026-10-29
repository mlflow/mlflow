import { useCallback, useMemo, useState } from 'react';
import {
  AggregationType,
  MetricViewType,
  TIME_BUCKET_DIMENSION_KEY,
  TraceDimensionKey,
  TraceMetricKey,
  type MetricDataPoint,
} from '@databricks/web-shared/model-trace-explorer';
import { ErrorCodes } from '../../../../common/constants';
import { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';
import { useMonitoringConfig } from '../../../hooks/useMonitoringConfig';
import { useSqlWarehouseContextSafe } from '../../experiment-page-tabs/SqlWarehouseContext';
import type { GenAIOverviewActivityPoint, GenAIOverviewTraceState } from '../genAIOverview.types';
import { generateTimeBuckets } from '../utils/chartUtils';
import { createGenAIOverviewTimeRange } from '../utils/timeUtils';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';

export const GENAI_OVERVIEW_TIME_INTERVAL_SECONDS = 24 * 60 * 60;
const FIRST_TRACE_POLL_INTERVAL_MS = 5000;

const isPermissionDenied = (error: unknown) =>
  error instanceof ErrorWrapper && error.getErrorCode() === ErrorCodes.PERMISSION_DENIED;

interface ResolveGenAIOverviewTraceStateParams {
  dataPoints?: MetricDataPoint[];
  isLoading: boolean;
  error: unknown;
  warehouseRequired: boolean;
  warehousesLoading: boolean;
  startTimeMs: number;
  endTimeMs: number;
}

export const resolveGenAIOverviewTraceState = ({
  dataPoints,
  isLoading,
  error,
  warehouseRequired,
  warehousesLoading,
  startTimeMs,
  endTimeMs,
}: ResolveGenAIOverviewTraceStateParams): GenAIOverviewTraceState => {
  if (warehousesLoading || isLoading) return { status: 'loading' };
  if (warehouseRequired) return { status: 'warehouse-required' };
  if (isPermissionDenied(error)) return { status: 'permission-denied' };
  if (error) return { status: 'error' };
  const countByTimestamp = new Map<number, number>();
  for (const dataPoint of dataPoints ?? []) {
    const timestampMs = new Date(dataPoint.dimensions?.[TIME_BUCKET_DIMENSION_KEY]).getTime();
    if (!Number.isNaN(timestampMs)) {
      countByTimestamp.set(
        timestampMs,
        (countByTimestamp.get(timestampMs) ?? 0) + (dataPoint.values?.[AggregationType.COUNT] ?? 0),
      );
    }
  }

  const activity: GenAIOverviewActivityPoint[] = generateTimeBuckets(
    startTimeMs,
    endTimeMs,
    GENAI_OVERVIEW_TIME_INTERVAL_SECONDS,
  ).map((timestampMs) => ({ timestampMs, count: countByTimestamp.get(timestampMs) ?? 0 }));
  const totalCount = activity.reduce((total, point) => total + point.count, 0);
  return totalCount > 0 ? { status: 'ready', activity, totalCount } : { status: 'empty', activity };
};

export const useGenAIOverviewTraceState = (experimentId: string, useRollingSevenDayRange: boolean = false) => {
  const { dateNow } = useMonitoringConfig();
  const [legacyTimeRange] = useState(() => createGenAIOverviewTimeRange(new Date(), false));
  const rollingTimeRange = useMemo(() => createGenAIOverviewTimeRange(dateNow, true), [dateNow]);
  const timeRange = useRollingSevenDayRange ? rollingTimeRange : legacyTimeRange;
  const sqlWarehouseContext = useSqlWarehouseContextSafe();
  const warehouseRequired = Boolean(sqlWarehouseContext?.hasV4Location && !sqlWarehouseContext.warehouseId);

  const { data, isLoading, error, refetch } = useTraceMetricsQuery({
    experimentIds: [experimentId],
    startTimeMs: timeRange.startTimeMs,
    endTimeMs: timeRange.endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    timeIntervalSeconds: GENAI_OVERVIEW_TIME_INTERVAL_SECONDS,
    dimensions: [TraceDimensionKey.TRACE_STATUS],
    enabled: !warehouseRequired,
    refetchIntervalWhileEmptyMs: FIRST_TRACE_POLL_INTERVAL_MS,
  });

  const state = useMemo(() => {
    return resolveGenAIOverviewTraceState({
      dataPoints: data?.data_points,
      isLoading,
      error,
      warehouseRequired,
      warehousesLoading: Boolean(sqlWarehouseContext?.warehousesLoading),
      startTimeMs: timeRange.startTimeMs,
      endTimeMs: timeRange.endTimeMs,
    });
  }, [data, error, isLoading, sqlWarehouseContext?.warehousesLoading, timeRange, warehouseRequired]);

  const retry = useCallback(() => {
    void refetch();
  }, [refetch]);

  return { state, retry };
};
