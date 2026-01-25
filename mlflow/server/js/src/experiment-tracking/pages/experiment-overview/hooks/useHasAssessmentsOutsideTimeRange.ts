import { useMemo } from 'react';
import {
  MetricViewType,
  AggregationType,
  AssessmentMetricKey,
  AssessmentFilterKey,
  AssessmentTypeValue,
  AssessmentDimensionKey,
  createAssessmentFilter,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { useOverviewChartContext } from '../OverviewChartContext';

/**
 * Hook that checks if there are assessments outside the current time range.
 * Makes a query without time filters to see if assessments exist at all.
 *
 * @returns Whether there are assessments outside the time range and loading state
 */
export function useHasAssessmentsOutsideTimeRange(enabled: boolean) {
  const { experimentId, filters: contextFilters } = useOverviewChartContext();

  // Filter for feedback assessments only, combined with context filters
  const filters = useMemo(
    () => [createAssessmentFilter(AssessmentFilterKey.TYPE, AssessmentTypeValue.FEEDBACK), ...(contextFilters || [])],
    [contextFilters],
  );

  // Query assessment counts WITHOUT time filters to see if any assessments exist
  const {
    data: countData,
    isLoading,
    error,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs: undefined,
    endTimeMs: undefined,
    viewType: MetricViewType.ASSESSMENTS,
    metricName: AssessmentMetricKey.ASSESSMENT_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    filters,
    dimensions: [AssessmentDimensionKey.ASSESSMENT_NAME],
    enabled,
  });

  const hasAssessments = useMemo(() => {
    if (!countData?.data_points) return false;
    return countData.data_points.length > 0;
  }, [countData?.data_points]);

  return {
    hasAssessmentsOutsideTimeRange: hasAssessments,
    isLoading,
    error,
  };
}
