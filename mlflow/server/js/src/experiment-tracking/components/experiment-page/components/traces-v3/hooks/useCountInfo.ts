import { useMemo } from 'react';
import {
  shouldUseInfinitePaginatedTraces,
  getEvalTabTotalTracesLimit,
} from '@databricks/web-shared/genai-traces-table';
import type { TableFilter } from '@databricks/web-shared/genai-traces-table';
import { useTraceMetricsQuery } from '../../../../../pages/experiment-overview/hooks/useTraceMetricsQuery';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  createTraceMetadataFilter,
  SESSION_ID_METADATA_KEY,
} from '@databricks/web-shared/model-trace-explorer';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { createMlflowSearchFilter } from '../../../../../../shared/web-shared/genai-traces-table/hooks/useMlflowTraces';

const getUniqueSessionCount = (traceInfos: ModelTraceInfoV3[] | undefined) =>
  new Set(
    (traceInfos ?? [])
      .map((traceInfo) => traceInfo.trace_metadata?.[SESSION_ID_METADATA_KEY])
      .filter((sessionId): sessionId is string => Boolean(sessionId)),
  ).size;

/**
 * Returns the countInfo object for the traces table toolbar badge.
 *
 * When infinite pagination is enabled, queries the trace metrics endpoint
 * to get the true total count of traces so the badge shows "X of Y" where
 * X = loaded traces and Y = total traces in the time period.
 *
 * When infinite pagination is disabled, falls back to the existing behavior
 * where totalCount comes from the metadata hook.
 */
export function useCountInfo({
  experimentIds,
  runUuid,
  timeRange,
  traceInfos,
  traceInfosCount,
  metadataTraceInfos,
  traceInfosLoading,
  metadataTotalCount,
  disabled,
  isGroupedBySession = false,
  additionalFilters,
}: {
  experimentIds: string[];
  runUuid?: string;
  timeRange?: { startTime?: string; endTime?: string };
  traceInfos?: ModelTraceInfoV3[];
  traceInfosCount?: number;
  metadataTraceInfos?: ModelTraceInfoV3[];
  traceInfosLoading: boolean;
  metadataTotalCount: number;
  disabled: boolean;
  isGroupedBySession?: boolean;
  additionalFilters?: TableFilter[];
}) {
  const usingInfinitePagination = shouldUseInfinitePaginatedTraces();
  const filters = useMemo(() => {
    const queryFilters: string[] = [];

    if (runUuid) {
      queryFilters.push(createTraceMetadataFilter('mlflow.sourceRun', runUuid));
    }

    if (additionalFilters?.length) {
      const additionalFilterQuery = createMlflowSearchFilter(undefined, undefined, additionalFilters);
      if (additionalFilterQuery) {
        queryFilters.push(additionalFilterQuery);
      }
    }

    return queryFilters.length > 0 ? queryFilters : undefined;
  }, [additionalFilters, runUuid]);

  const startTimeMs = timeRange?.startTime ? Number(timeRange.startTime) : undefined;
  const endTimeMs = timeRange?.endTime ? Number(timeRange.endTime) : undefined;

  const { data: traceCountMetrics, isLoading: traceCountLoading } = useTraceMetricsQuery({
    experimentIds,
    viewType: MetricViewType.TRACES,
    metricName: isGroupedBySession ? TraceMetricKey.SESSION_COUNT : TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    startTimeMs,
    endTimeMs,
    enabled: (usingInfinitePagination || isGroupedBySession) && !disabled,
    filters,
  });
  const metricsTotal = traceCountMetrics?.data_points?.[0]?.values?.[AggregationType.COUNT];
  const currentCount = isGroupedBySession
    ? getUniqueSessionCount(traceInfos)
    : traceInfos?.length ?? traceInfosCount ?? 0;
  const filteredTotalCount = isGroupedBySession ? getUniqueSessionCount(metadataTraceInfos) : metadataTotalCount;

  return useMemo(() => {
    if (usingInfinitePagination || isGroupedBySession) {
      return {
        currentCount,
        logCountLoading: traceInfosLoading || traceCountLoading,
        totalCount: metricsTotal ?? filteredTotalCount ?? currentCount ?? 0,
        maxAllowedCount: Infinity,
      };
    }
    return {
      currentCount,
      logCountLoading: traceInfosLoading,
      totalCount: filteredTotalCount,
      maxAllowedCount: getEvalTabTotalTracesLimit(),
    };
  }, [
    usingInfinitePagination,
    currentCount,
    filteredTotalCount,
    traceInfosLoading,
    traceCountLoading,
    metricsTotal,
    isGroupedBySession,
  ]);
}
