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

export interface UseAssessmentChartsSectionDataResult {
  /** Sorted list of assessment names */
  assessmentNames: string[];
  /** Map of assessment name to its average value (only for numeric assessments) */
  avgValuesByName: Map<string, number>;
  /** Map of assessment name to its total count */
  countsByName: Map<string, number>;
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there are any assessments */
  hasData: boolean;
}

/**
 * Custom hook that fetches and processes assessment data for the charts section.
 * Queries assessments grouped by name using COUNT to get all assessments (including string types),
 * and also fetches AVG values for numeric assessments.
 * Uses OverviewChartContext to get chart props.
 *
 * @returns Assessment names, average values (for numeric only), loading state, and error state
 */
export function useAssessmentChartsSectionData(): UseAssessmentChartsSectionDataResult {
  const { experimentIds, startTimeMs, endTimeMs, filters: contextFilters } = useOverviewChartContext();
  // Filter for feedback assessments only, combined with context filters
  const filters = useMemo(
    () => [createAssessmentFilter(AssessmentFilterKey.TYPE, AssessmentTypeValue.FEEDBACK), ...(contextFilters || [])],
    [contextFilters],
  );

  // Query assessment counts grouped by name to get ALL assessments
  const {
    data: countData,
    isLoading: isLoadingCount,
    error: countError,
  } = useTraceMetricsQuery({
    experimentIds,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.ASSESSMENTS,
    metricName: AssessmentMetricKey.ASSESSMENT_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    filters,
    dimensions: [AssessmentDimensionKey.ASSESSMENT_NAME],
  });

  // Query average values grouped by name (only numeric assessments will have values)
  const {
    data: avgData,
    isLoading: isLoadingAvg,
    error: avgError,
  } = useTraceMetricsQuery({
    experimentIds,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.ASSESSMENTS,
    metricName: AssessmentMetricKey.ASSESSMENT_VALUE,
    aggregations: [{ aggregation_type: AggregationType.AVG }],
    filters,
    dimensions: [AssessmentDimensionKey.ASSESSMENT_NAME],
  });

  // Extract assessment names and counts from count query
  const { assessmentNames, countsByName } = useMemo(() => {
    if (!countData?.data_points) return { assessmentNames: [], countsByName: new Map<string, number>() };

    const names = new Set<string>();
    const counts = new Map<string, number>();
    for (const dp of countData.data_points) {
      const name = dp.dimensions?.[AssessmentDimensionKey.ASSESSMENT_NAME];
      const count = dp.values?.[AggregationType.COUNT];
      if (name && count !== undefined) {
        names.add(name);
        counts.set(name, count);
      }
    }
    return { assessmentNames: Array.from(names).sort(), countsByName: counts };
  }, [countData?.data_points]);

  // Extract average values from avg query (only numeric assessments)
  const avgValuesByName = useMemo(() => {
    if (!avgData?.data_points) return new Map<string, number>();

    const avgValues = new Map<string, number>();
    for (const dp of avgData.data_points) {
      const name = dp.dimensions?.[AssessmentDimensionKey.ASSESSMENT_NAME];
      const avgValue = dp.values?.[AggregationType.AVG];
      if (name && avgValue !== undefined) {
        avgValues.set(name, avgValue);
      }
    }
    return avgValues;
  }, [avgData?.data_points]);

  const isLoading = isLoadingCount || isLoadingAvg;
  const error = countError || avgError;

  return {
    assessmentNames,
    avgValuesByName,
    countsByName,
    isLoading,
    error,
    hasData: assessmentNames.length > 0,
  };
}
