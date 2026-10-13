import { useMemo } from 'react';
import {
  MetricViewType,
  AggregationType,
  AssessmentMetricKey,
  AssessmentFilterKey,
  AssessmentTypeValue,
  AssessmentDimensionKey,
  TIME_BUCKET_DIMENSION_KEY,
  createAssessmentFilter,
  INTERNAL_ASSESSMENT_ISSUE_DISCOVERY_JUDGE,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { useOverviewChartContext } from '../OverviewChartContext';
import { formatTimestampForTraceMetrics } from '../utils/chartUtils';
import {
  sortValuesAlphanumerically,
  shouldCreateHistogramBuckets,
  createHistogramBuckets,
  findBucketIndexForValue,
} from '../utils/distributionUtils';

export interface AssessmentChartDataPoint {
  name: string;
  value: number | null;
  timestampMs: number;
}

export interface DistributionChartDataPoint {
  name: string;
  count: number;
}

export interface UseAssessmentChartsSectionDataResult {
  assessmentNames: string[];
  avgValuesByName: Map<string, number>;
  countsByName: Map<string, number>;
  timeSeriesChartDataByName: Map<string, AssessmentChartDataPoint[]>;
  distributionChartDataByName: Map<string, DistributionChartDataPoint[]>;
  isLoading: boolean;
  error: unknown;
  hasData: boolean;
}

export function useAssessmentChartsSectionData(): UseAssessmentChartsSectionDataResult {
  const { experimentIds, startTimeMs, endTimeMs, timeIntervalSeconds, timeBuckets } = useOverviewChartContext();
  const filters = useMemo(() => [createAssessmentFilter(AssessmentFilterKey.TYPE, AssessmentTypeValue.FEEDBACK)], []);

  // Single time-series query for all assessments, grouped by assessment name
  const {
    data: timeSeriesData,
    isLoading: isLoadingTimeSeries,
    error: timeSeriesError,
  } = useTraceMetricsQuery({
    experimentIds,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.ASSESSMENTS,
    metricName: AssessmentMetricKey.ASSESSMENT_VALUE,
    aggregations: [{ aggregation_type: AggregationType.AVG }],
    dimensions: [AssessmentDimensionKey.ASSESSMENT_NAME],
    timeIntervalSeconds,
    filters,
  });

  // Single distribution query for all assessments, grouped by assessment name and value
  const {
    data: distributionData,
    isLoading: isLoadingDistribution,
    error: distributionError,
  } = useTraceMetricsQuery({
    experimentIds,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.ASSESSMENTS,
    metricName: AssessmentMetricKey.ASSESSMENT_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    dimensions: [AssessmentDimensionKey.ASSESSMENT_NAME, AssessmentDimensionKey.ASSESSMENT_VALUE],
    filters,
  });

  // Derive assessment names, counts, and averages from the distribution query
  const { assessmentNames, countsByName, avgValuesByName } = useMemo(() => {
    const distributionPoints = distributionData?.data_points ?? [];

    const counts = new Map<string, number>();
    const valuesByName = new Map<string, Map<string, number>>();

    for (const dp of distributionPoints) {
      const name = dp.dimensions?.[AssessmentDimensionKey.ASSESSMENT_NAME];
      const value = dp.dimensions?.[AssessmentDimensionKey.ASSESSMENT_VALUE];
      const count = dp.values?.[AggregationType.COUNT];
      if (!name || name === INTERNAL_ASSESSMENT_ISSUE_DISCOVERY_JUDGE || value === undefined || count === undefined) {
        continue;
      }
      counts.set(name, (counts.get(name) ?? 0) + count);
      let nameValues = valuesByName.get(name);
      if (!nameValues) {
        nameValues = new Map();
        valuesByName.set(name, nameValues);
      }
      nameValues.set(value, (nameValues.get(value) ?? 0) + count);
    }

    const avgValues = new Map<string, number>();
    for (const [name, values] of valuesByName) {
      let weightedSum = 0;
      let totalCount = 0;
      let allNumeric = true;
      for (const [value, count] of values) {
        const numValue = parseFloat(value);
        if (isNaN(numValue)) {
          allNumeric = false;
          break;
        }
        weightedSum += numValue * count;
        totalCount += count;
      }
      if (allNumeric && totalCount > 0) {
        avgValues.set(name, weightedSum / totalCount);
      }
    }

    const names = Array.from(counts.keys()).sort();
    return { assessmentNames: names, countsByName: counts, avgValuesByName: avgValues };
  }, [distributionData?.data_points]);

  const timeSeriesChartDataByName = useMemo(() => {
    const result = new Map<string, AssessmentChartDataPoint[]>();
    const timeSeriesPoints = timeSeriesData?.data_points ?? [];

    const valuesByNameAndTime = new Map<string, Map<number, number | null>>();
    for (const dp of timeSeriesPoints) {
      const name = dp.dimensions?.[AssessmentDimensionKey.ASSESSMENT_NAME];
      const timeBucket = dp.dimensions?.[TIME_BUCKET_DIMENSION_KEY];
      const avgValue = dp.values?.[AggregationType.AVG];
      if (name && timeBucket) {
        let timeMap = valuesByNameAndTime.get(name);
        if (!timeMap) {
          timeMap = new Map();
          valuesByNameAndTime.set(name, timeMap);
        }
        timeMap.set(new Date(timeBucket).getTime(), avgValue ?? null);
      }
    }

    for (const name of assessmentNames) {
      const valuesByTime = valuesByNameAndTime.get(name);
      result.set(
        name,
        timeBuckets.map((timestampMs) => ({
          name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
          value: valuesByTime?.get(timestampMs) ?? null,
          timestampMs,
        })),
      );
    }

    return result;
  }, [timeSeriesData?.data_points, assessmentNames, timeBuckets, timeIntervalSeconds]);

  const distributionChartDataByName = useMemo(() => {
    const result = new Map<string, DistributionChartDataPoint[]>();
    const distributionPoints = distributionData?.data_points ?? [];

    const valueCountsByName = new Map<string, Record<string, number>>();
    for (const dp of distributionPoints) {
      const name = dp.dimensions?.[AssessmentDimensionKey.ASSESSMENT_NAME];
      const rawValue = dp.dimensions?.[AssessmentDimensionKey.ASSESSMENT_VALUE];
      const count = dp.values?.[AggregationType.COUNT];
      if (name && rawValue !== undefined) {
        let valueCounts = valueCountsByName.get(name);
        if (!valueCounts) {
          valueCounts = {};
          valueCountsByName.set(name, valueCounts);
        }
        valueCounts[rawValue] = (valueCounts[rawValue] ?? 0) + (count ?? 0);
      }
    }

    for (const name of assessmentNames) {
      const valueCounts = valueCountsByName.get(name) ?? {};
      const allValues = Object.keys(valueCounts);

      if (shouldCreateHistogramBuckets(allValues)) {
        const buckets = createHistogramBuckets(allValues);
        const bucketCounts = buckets.map(() => 0);

        for (const [value, count] of Object.entries(valueCounts)) {
          const numValue = parseFloat(value);
          if (!isNaN(numValue)) {
            bucketCounts[findBucketIndexForValue(numValue, buckets)] += count;
          }
        }

        result.set(
          name,
          buckets.map((bucket, index) => ({
            name: bucket.label,
            count: bucketCounts[index],
          })),
        );
      } else {
        const sortedValues = sortValuesAlphanumerically(allValues);
        result.set(
          name,
          sortedValues.map((value) => ({
            name: value,
            count: valueCounts[value] ?? 0,
          })),
        );
      }
    }

    return result;
  }, [distributionData?.data_points, assessmentNames]);

  const isLoading = isLoadingTimeSeries || isLoadingDistribution;
  const error = timeSeriesError || distributionError;

  return {
    assessmentNames,
    avgValuesByName,
    countsByName,
    timeSeriesChartDataByName,
    distributionChartDataByName,
    isLoading,
    error,
    hasData: assessmentNames.length > 0,
  };
}
