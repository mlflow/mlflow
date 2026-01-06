import { useMemo } from 'react';
import {
  MetricViewType,
  AggregationType,
  AssessmentMetricKey,
  AssessmentFilterKey,
  AssessmentType,
  AssessmentDimensionKey,
  createAssessmentFilter,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import type { OverviewChartProps } from '../types';

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
 *
 * @param props - Chart props including experimentId and time range
 * @returns Assessment names, average values, loading state, and error state
 */
export function useAssessmentChartsSectionData({
  experimentId,
  startTimeMs,
  endTimeMs,
}: Pick<OverviewChartProps, 'experimentId' | 'startTimeMs' | 'endTimeMs'>): UseAssessmentChartsSectionDataResult {
  // Filter for feedback assessments only
  const filters = useMemo(() => [createAssessmentFilter(AssessmentFilterKey.TYPE, AssessmentType.FEEDBACK)], []);

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

    const names: string[] = [];
    const avgValues = new Map<string, number>();

    for (const dp of data.data_points) {
      const name = dp.dimensions?.[AssessmentDimensionKey.ASSESSMENT_NAME];
      if (name) {
        names.push(name);
        const avgValue = dp.values?.[AggregationType.AVG];
        if (avgValue !== undefined) {
          avgValues.set(name, avgValue);
        }
      }
    }

    return { assessmentNames: names.sort(), avgValuesByName: avgValues };
  }, [data?.data_points]);

  return {
    assessmentNames,
    avgValuesByName,
    isLoading,
    error,
    hasData: assessmentNames.length > 0,
  };
}
