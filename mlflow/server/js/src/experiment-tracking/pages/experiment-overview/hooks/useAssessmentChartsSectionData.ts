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
  /** Map of assessment name to its average value */
  avgValuesByName: Map<string, number>;
  /** Whether data is currently being fetched */
  isLoading: boolean;
  /** Error if data fetching failed */
  error: unknown;
  /** Whether there are any assessments */
  hasData: boolean;
}

/**
 * Custom hook that fetches and processes assessment data for the charts section.
 * Queries assessments grouped by name and extracts their average values.
 * Uses OverviewChartContext to get chart props.
 *
 * @returns Assessment names, average values, loading state, and error state
 */
export function useAssessmentChartsSectionData(): UseAssessmentChartsSectionDataResult {
  const { experimentId, startTimeMs, endTimeMs } = useOverviewChartContext();
  // Filter for feedback assessments only
  const filters = useMemo(() => [createAssessmentFilter(AssessmentFilterKey.TYPE, AssessmentTypeValue.FEEDBACK)], []);

  // Query assessments grouped by assessment_name to get the list and average values
  const { data, isLoading, error } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.ASSESSMENTS,
    metricName: AssessmentMetricKey.ASSESSMENT_VALUE,
    aggregations: [{ aggregation_type: AggregationType.AVG }],
    filters,
    dimensions: [AssessmentDimensionKey.ASSESSMENT_NAME],
  });

  // Extract assessment names and their average values from the response
  const { assessmentNames, avgValuesByName } = useMemo(() => {
    if (!data?.data_points) return { assessmentNames: [], avgValuesByName: new Map<string, number>() };

    const avgValues = new Map<string, number>();

    for (const dp of data.data_points) {
      const name = dp.dimensions?.[AssessmentDimensionKey.ASSESSMENT_NAME];
      const avgValue = dp.values?.[AggregationType.AVG];
      if (name && avgValue !== undefined) {
        avgValues.set(name, avgValue);
      }
    }

    return { assessmentNames: Array.from(avgValues.keys()).sort(), avgValuesByName: avgValues };
  }, [data?.data_points]);

  return {
    assessmentNames,
    avgValuesByName,
    isLoading,
    error,
    hasData: assessmentNames.length > 0,
  };
}
