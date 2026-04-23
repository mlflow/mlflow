import { useMemo } from 'react';
import { shouldUseInfinitePaginatedTraces } from '@databricks/web-shared/genai-traces-table';
import type { AssessmentCountMetrics } from '@databricks/web-shared/genai-traces-table';
import { useTraceMetricsQuery } from '../../../../../pages/experiment-overview/hooks/useTraceMetricsQuery';
import {
  MetricViewType,
  AggregationType,
  AssessmentMetricKey,
  AssessmentDimensionKey,
  createTraceMetadataFilter,
} from '@databricks/web-shared/model-trace-explorer';

/**
 * Fetches categorical assessment value distributions from the trace metrics API.
 *
 * When infinite pagination is enabled, the client only has a subset of traces
 * loaded. This hook queries the metrics endpoint to get accurate value counts
 * (e.g. pass/fail distributions) across all matching traces.
 *
 * Returns undefined when the flag is off so callers fall back to client-side aggregation.
 */
export function useAssessmentCountMetrics({
  experimentIds,
  runUuid,
  timeRange,
  disabled,
}: {
  experimentIds: string[];
  runUuid?: string;
  timeRange?: { startTime?: string; endTime?: string };
  disabled: boolean;
}): AssessmentCountMetrics | undefined {
  const usingInfinitePagination = shouldUseInfinitePaginatedTraces();
  const enabled = usingInfinitePagination && !disabled;

  const filters = useMemo(
    () => (runUuid ? [createTraceMetadataFilter('mlflow.sourceRun', runUuid)] : undefined),
    [runUuid],
  );

  const startTimeMs = timeRange?.startTime ? Number(timeRange.startTime) : undefined;
  const endTimeMs = timeRange?.endTime ? Number(timeRange.endTime) : undefined;

  const { data, isLoading } = useTraceMetricsQuery({
    experimentIds,
    viewType: MetricViewType.ASSESSMENTS,
    metricName: AssessmentMetricKey.ASSESSMENT_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    dimensions: [AssessmentDimensionKey.ASSESSMENT_NAME, AssessmentDimensionKey.ASSESSMENT_VALUE],
    startTimeMs,
    endTimeMs,
    enabled,
    filters,
  });

  return useMemo(() => {
    if (!enabled) return undefined;

    const metrics =
      data?.data_points?.map((dp) => ({
        assessmentName: dp.dimensions[AssessmentDimensionKey.ASSESSMENT_NAME],
        assessmentValue: dp.dimensions[AssessmentDimensionKey.ASSESSMENT_VALUE],
        count: dp.values[AggregationType.COUNT] ?? 0,
      })) ?? [];

    return { data: metrics, isLoading };
  }, [enabled, data, isLoading]);
}
