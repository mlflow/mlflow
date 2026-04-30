import { useMemo } from 'react';
import {
  shouldUseInfinitePaginatedTraces,
  getEvalTabTotalTracesLimit,
} from '@databricks/web-shared/genai-traces-table';
import type { TableFilter } from '@databricks/web-shared/genai-traces-table';
import { FilterOperator, TracesTableColumnGroup } from '@databricks/web-shared/genai-traces-table';
import { useTraceMetricsQuery } from '../../../../../pages/experiment-overview/hooks/useTraceMetricsQuery';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  TraceFilterKey,
  TRACE_FILTER_VIEW_TYPE,
  createTraceMetadataFilter,
  SESSION_ID_METADATA_KEY,
} from '@databricks/web-shared/model-trace-explorer';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import {
  CUSTOM_METADATA_COLUMN_ID,
  ISSUE_ID_COLUMN_ID,
  LOGGED_MODEL_COLUMN_ID,
  RUN_NAME_COLUMN_ID,
  SESSION_COLUMN_ID,
  SOURCE_COLUMN_ID,
  SPAN_NAME_COLUMN_ID,
  SPAN_STATUS_COLUMN_ID,
  SPAN_TYPE_COLUMN_ID,
  STATE_COLUMN_ID,
  TRACE_NAME_COLUMN_ID,
  USER_COLUMN_ID,
} from '../../../../../../shared/web-shared/genai-traces-table/hooks/useTableColumns';
import { getCustomMetadataKeyFromColumnId } from '../../../../../../shared/web-shared/genai-traces-table/utils/TraceUtils';

const getUniqueSessionCount = (traceInfos: ModelTraceInfoV3[] | undefined) =>
  new Set(
    (traceInfos ?? [])
      .map((traceInfo) => traceInfo.trace_metadata?.[SESSION_ID_METADATA_KEY])
      .filter((sessionId): sessionId is string => Boolean(sessionId)),
  ).size;

const quoteFilterValue = (value: string | number | boolean) => `"${String(value).replace(/"/g, '\\"')}"`;

const createTraceMetricsClause = (
  identifier: string,
  operator: TableFilter['operator'],
  value?: string | number | boolean,
) => {
  if (operator === FilterOperator.IS_NULL || operator === FilterOperator.IS_NOT_NULL) {
    return `${identifier} ${operator}`;
  }
  if (value === undefined) {
    return undefined;
  }
  return `${identifier} ${operator} ${quoteFilterValue(value)}`;
};

const createTraceTagFilter = (tagKey: string, operator: TableFilter['operator'], value?: string | number | boolean) =>
  createTraceMetricsClause(
    `${TRACE_FILTER_VIEW_TYPE}.${TraceFilterKey.TAG}.\`${tagKey}\``,
    operator,
    value,
  );

const createTraceMetadataClause = (
  metadataKey: string,
  operator: TableFilter['operator'],
  value?: string | number | boolean,
) => createTraceMetricsClause(`${TRACE_FILTER_VIEW_TYPE}.${TraceFilterKey.METADATA}.\`${metadataKey}\``, operator, value);

const UNSUPPORTED_TRACE_METRICS_FILTER = Symbol('UNSUPPORTED_TRACE_METRICS_FILTER');

const convertToTraceMetricsFilter = (
  filter: TableFilter,
): string | typeof UNSUPPORTED_TRACE_METRICS_FILTER | undefined => {
  switch (filter.column) {
    case STATE_COLUMN_ID:
      return createTraceMetricsClause(`${TRACE_FILTER_VIEW_TYPE}.${TraceFilterKey.STATUS}`, filter.operator, filter.value);
    case USER_COLUMN_ID:
      if (filter.operator === FilterOperator.CONTAINS) {
        return UNSUPPORTED_TRACE_METRICS_FILTER;
      }
      return createTraceMetadataClause('mlflow.trace.user', filter.operator, filter.value);
    case SESSION_COLUMN_ID:
      if (filter.operator === FilterOperator.CONTAINS) {
        return UNSUPPORTED_TRACE_METRICS_FILTER;
      }
      return createTraceMetadataClause(SESSION_ID_METADATA_KEY, filter.operator, filter.value);
    case RUN_NAME_COLUMN_ID:
      if (filter.operator === FilterOperator.CONTAINS) {
        return UNSUPPORTED_TRACE_METRICS_FILTER;
      }
      return createTraceMetadataClause('mlflow.sourceRun', filter.operator, filter.value);
    case LOGGED_MODEL_COLUMN_ID:
      if (filter.operator === FilterOperator.CONTAINS) {
        return UNSUPPORTED_TRACE_METRICS_FILTER;
      }
      return createTraceMetadataClause('mlflow.modelId', filter.operator, filter.value);
    case SOURCE_COLUMN_ID:
      if (filter.operator === FilterOperator.CONTAINS) {
        return UNSUPPORTED_TRACE_METRICS_FILTER;
      }
      return createTraceMetadataClause('mlflow.source.name', filter.operator, filter.value);
    case SPAN_NAME_COLUMN_ID:
      if (filter.operator === FilterOperator.CONTAINS) {
        return UNSUPPORTED_TRACE_METRICS_FILTER;
      }
      return createTraceMetricsClause('span.name', filter.operator, filter.value);
    case SPAN_TYPE_COLUMN_ID:
      if (filter.operator === FilterOperator.CONTAINS) {
        return UNSUPPORTED_TRACE_METRICS_FILTER;
      }
      return createTraceMetricsClause('span.type', filter.operator, filter.value);
    case SPAN_STATUS_COLUMN_ID:
      return createTraceMetricsClause('span.status', filter.operator, filter.value);
    case ISSUE_ID_COLUMN_ID:
      return createTraceMetricsClause('assessment.name', filter.operator, filter.value);
    case TRACE_NAME_COLUMN_ID:
      return UNSUPPORTED_TRACE_METRICS_FILTER;
    case TracesTableColumnGroup.TAG:
      if (!filter.key) {
        return undefined;
      }
      return createTraceTagFilter(filter.key, filter.operator, filter.value);
    default:
      if (typeof filter.column === 'string' && filter.column.startsWith(CUSTOM_METADATA_COLUMN_ID)) {
        if (filter.operator === FilterOperator.CONTAINS) {
          return UNSUPPORTED_TRACE_METRICS_FILTER;
        }
        return createTraceMetadataClause(getCustomMetadataKeyFromColumnId(filter.column), filter.operator, filter.value);
      }
      return UNSUPPORTED_TRACE_METRICS_FILTER;
  }
};

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
  const { filters, hasUnsupportedFilters } = useMemo(() => {
    const queryFilters: string[] = [];
    let hasUnsupported = false;

    if (runUuid) {
      queryFilters.push(createTraceMetadataFilter('mlflow.sourceRun', runUuid));
    }

    if (additionalFilters?.length) {
      additionalFilters.forEach((filter) => {
        const metricsFilter = convertToTraceMetricsFilter(filter);
        if (metricsFilter === UNSUPPORTED_TRACE_METRICS_FILTER) {
          hasUnsupported = true;
          return;
        }
        if (metricsFilter) {
          queryFilters.push(metricsFilter);
        }
      });
    }

    return {
      filters: queryFilters.length > 0 ? queryFilters : undefined,
      hasUnsupportedFilters: hasUnsupported,
    };
  }, [additionalFilters, runUuid]);

  // Grouped session totals still depend on backend support for the `session_count`
  // metric, so keep the frontend split on the safe metadata fallback path until
  // that metric is available.
  const shouldQueryTraceMetrics =
    usingInfinitePagination && !isGroupedBySession && !disabled && !hasUnsupportedFilters;

  const startTimeMs = timeRange?.startTime ? Number(timeRange.startTime) : undefined;
  const endTimeMs = timeRange?.endTime ? Number(timeRange.endTime) : undefined;

  const { data: traceCountMetrics, isLoading: traceCountLoading } = useTraceMetricsQuery({
    experimentIds,
    viewType: MetricViewType.TRACES,
    metricName: isGroupedBySession ? TraceMetricKey.SESSION_COUNT : TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    startTimeMs,
    endTimeMs,
    enabled: shouldQueryTraceMetrics,
    filters,
  });
  const metricsTotal = shouldQueryTraceMetrics
    ? traceCountMetrics?.data_points?.[0]?.values?.[AggregationType.COUNT]
    : undefined;
  const currentCount = isGroupedBySession
    ? getUniqueSessionCount(traceInfos)
    : traceInfos?.length ?? traceInfosCount ?? 0;
  const filteredTotalCount = isGroupedBySession ? getUniqueSessionCount(metadataTraceInfos) : metadataTotalCount;

  return useMemo(() => {
    if (usingInfinitePagination || isGroupedBySession) {
      const fallbackTotal = Math.max(filteredTotalCount ?? 0, currentCount ?? 0);
      return {
        currentCount,
        logCountLoading: traceInfosLoading || traceCountLoading,
        totalCount: metricsTotal ?? fallbackTotal,
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
