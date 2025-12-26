import React, { useMemo, useCallback } from 'react';
import { CheckCircleIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from 'recharts';
import {
  MetricViewType,
  AggregationType,
  AssessmentMetricKey,
  AssessmentFilterKey,
  AssessmentDimensionKey,
  createAssessmentFilter,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from '../hooks/useTraceMetricsQuery';
import {
  ChartLoadingState,
  ChartErrorState,
  ChartEmptyState,
  ChartHeader,
  ChartContainer,
  useChartTooltipStyle,
  useChartXAxisProps,
  useChartLegendFormatter,
} from './ChartCardWrapper';
import { formatTimestampForTraceMetrics, useTimestampValueMap } from '../utils/chartUtils';
import type { OverviewChartProps } from '../types';

/**
 * Sort assessment values intelligently.
 * Numbers are sorted numerically, strings alphabetically.
 */
const sortAssessmentValues = (values: string[]): string[] =>
  [...values].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

/**
 * Determine if values should be bucketed into ranges.
 * Returns true if:
 * - Values are floats (have decimals), OR
 * - Values are integers with more than 5 unique values
 */
const shouldBucketValues = (values: string[]): boolean => {
  if (values.length === 0) return false;

  const numericValues = values.map((v) => parseFloat(v)).filter((n) => !isNaN(n));
  // Most values should be numeric
  if (numericValues.length < values.length * 0.8) return false;

  const uniqueValues = new Set(numericValues);
  const hasDecimals = numericValues.some((n) => !Number.isInteger(n));

  // Bucket if: values are floats, OR integers with more than 5 unique values
  return hasDecimals || uniqueValues.size > 5;
};

/**
 * Create buckets for continuous numeric values.
 * Returns bucket definitions based on min/max of the data.
 */
const createBuckets = (values: string[], numBuckets = 5): { min: number; max: number; label: string }[] => {
  const numericValues = values.map((v) => parseFloat(v)).filter((n) => !isNaN(n));
  if (numericValues.length === 0) return [];

  const min = Math.min(...numericValues);
  const max = Math.max(...numericValues);
  const range = max - min;
  const bucketSize = range / numBuckets;

  return Array.from({ length: numBuckets }, (_, i) => {
    const bucketMin = min + i * bucketSize;
    const bucketMax = i === numBuckets - 1 ? max : min + (i + 1) * bucketSize;
    return {
      min: bucketMin,
      max: bucketMax,
      label: `${bucketMin.toFixed(2)}-${bucketMax.toFixed(2)}`,
    };
  });
};

/**
 * Get the bucket index for a value.
 */
const getBucketIndex = (value: number, buckets: { min: number; max: number }[]): number => {
  for (let i = 0; i < buckets.length; i++) {
    if (value >= buckets[i].min && value <= buckets[i].max) {
      return i;
    }
  }
  return buckets.length - 1; // Default to last bucket
};

/** Local component for chart panel with label */
const ChartPanel: React.FC<{ label: React.ReactNode; children: React.ReactElement }> = ({ label, children }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ flex: 1 }}>
      <Typography.Text color="secondary" size="sm">
        {label}
      </Typography.Text>
      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export interface TraceAssessmentChartProps extends OverviewChartProps {
  /** The name of the assessment to display (e.g., "Correctness", "Relevance") */
  assessmentName: string;
  /** Optional color for the line chart. Defaults to green. */
  lineColor?: string;
  /** Optional pre-computed average value (to avoid redundant queries) */
  avgValue?: number;
}

export const TraceAssessmentChart: React.FC<TraceAssessmentChartProps> = ({
  experimentId,
  startTimeMs,
  endTimeMs,
  timeIntervalSeconds,
  timeBuckets,
  assessmentName,
  lineColor,
  avgValue,
}) => {
  const { theme } = useDesignSystemTheme();
  const tooltipStyle = useChartTooltipStyle();
  const xAxisProps = useChartXAxisProps();
  const legendFormatter = useChartLegendFormatter();

  // Use provided color or default to green
  const chartLineColor = lineColor || theme.colors.green500;

  // Create filters for feedback assessments with the given name
  const filters = useMemo(() => [createAssessmentFilter(AssessmentFilterKey.NAME, assessmentName)], [assessmentName]);

  // Fetch assessment values over time for the line chart
  const {
    data: timeSeriesData,
    isLoading: isLoadingTimeSeries,
    error: timeSeriesError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.ASSESSMENTS,
    metricName: AssessmentMetricKey.ASSESSMENT_VALUE,
    aggregations: [{ aggregation_type: AggregationType.AVG }],
    filters,
    timeIntervalSeconds,
  });

  // Fetch assessment counts grouped by assessment_value for the bar chart
  const {
    data: distributionData,
    isLoading: isLoadingDistribution,
    error: distributionError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.ASSESSMENTS,
    metricName: AssessmentMetricKey.ASSESSMENT_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    filters,
    dimensions: [AssessmentDimensionKey.ASSESSMENT_VALUE],
  });

  const timeSeriesDataPoints = useMemo(() => timeSeriesData?.data_points || [], [timeSeriesData?.data_points]);
  const distributionDataPoints = useMemo(() => distributionData?.data_points || [], [distributionData?.data_points]);

  // Create a map of values by timestamp for the line chart
  const valueExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => dp.values?.[AggregationType.AVG] || 0,
    [],
  );
  const valuesByTimestamp = useTimestampValueMap(timeSeriesDataPoints, valueExtractor);

  // Prepare time series chart data - fill in all time buckets with 0 for missing data
  const timeSeriesChartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => {
      const value = valuesByTimestamp.get(timestampMs);
      return {
        name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
        value: value || 0,
      };
    });
  }, [timeBuckets, valuesByTimestamp, timeIntervalSeconds]);

  // Prepare distribution chart data - use actual values from API
  const distributionChartData = useMemo(() => {
    // Collect raw counts by assessment value
    const valueCounts: Record<string, number> = {};

    for (const dp of distributionDataPoints) {
      const rawValue = dp.dimensions?.[AssessmentDimensionKey.ASSESSMENT_VALUE];
      if (rawValue !== undefined) {
        const count = dp.values?.[AggregationType.COUNT] || 0;
        valueCounts[rawValue] = (valueCounts[rawValue] || 0) + count;
      }
    }

    const allValues = Object.keys(valueCounts);

    // Check if we should bucket float values
    if (shouldBucketValues(allValues)) {
      const buckets = createBuckets(allValues);
      const bucketCounts = buckets.map(() => 0);

      // Aggregate counts into buckets
      for (const [value, count] of Object.entries(valueCounts)) {
        const numValue = parseFloat(value);
        if (!isNaN(numValue)) {
          const bucketIndex = getBucketIndex(numValue, buckets);
          bucketCounts[bucketIndex] += count;
        }
      }

      return buckets.map((bucket, index) => ({
        name: bucket.label,
        count: bucketCounts[index],
      }));
    }

    // For non-float values (integers, strings, booleans), use as-is
    const sortedValues = sortAssessmentValues(allValues);
    return sortedValues.map((value) => ({
      name: value,
      count: valueCounts[value] || 0,
    }));
  }, [distributionDataPoints]);

  const isLoading = isLoadingTimeSeries || isLoadingDistribution;
  const error = timeSeriesError || distributionError;
  const hasData = timeSeriesDataPoints.length > 0 || distributionDataPoints.length > 0;

  if (isLoading) {
    return <ChartLoadingState />;
  }

  if (error) {
    return <ChartErrorState />;
  }

  if (!hasData) {
    return (
      <ChartContainer>
        <ChartHeader icon={<CheckCircleIcon css={{ color: chartLineColor }} />} title={assessmentName} />
        <ChartEmptyState />
      </ChartContainer>
    );
  }

  return (
    <ChartContainer>
      <ChartHeader
        icon={<CheckCircleIcon css={{ color: chartLineColor }} />}
        title={assessmentName}
        value={avgValue !== undefined ? avgValue.toFixed(2) : undefined}
        subtitle={
          avgValue !== undefined ? (
            <FormattedMessage defaultMessage="avg score" description="Subtitle for average assessment score" />
          ) : undefined
        }
      />

      {/* Two charts side by side */}
      <div css={{ display: 'flex', gap: theme.spacing.lg, marginTop: theme.spacing.sm }}>
        {/* Left: Distribution bar chart */}
        <ChartPanel
          label={
            <FormattedMessage
              defaultMessage="Total aggregate scores"
              description="Label for assessment score distribution chart"
            />
          }
        >
          <BarChart data={distributionChartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
            <XAxis dataKey="name" {...xAxisProps} />
            <YAxis allowDecimals={false} {...xAxisProps} />
            <Tooltip
              contentStyle={tooltipStyle}
              cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
              formatter={(value: number) => [value, 'count']}
            />
            <Legend formatter={legendFormatter} />
            <Bar dataKey="count" fill={chartLineColor} radius={[4, 4, 0, 0]} />
          </BarChart>
        </ChartPanel>

        {/* Right: Time series line chart */}
        <ChartPanel
          label={
            <FormattedMessage
              defaultMessage="Moving average over time"
              description="Label for assessment score over time chart"
            />
          }
        >
          <LineChart data={timeSeriesChartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
            <XAxis dataKey="name" {...xAxisProps} />
            <YAxis hide />
            <Tooltip
              contentStyle={tooltipStyle}
              cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
              formatter={(value: number) => [value.toFixed(2), assessmentName]}
            />
            <Legend formatter={legendFormatter} />
            <Line
              type="monotone"
              dataKey="value"
              name={assessmentName}
              stroke={chartLineColor}
              strokeWidth={2}
              dot={false}
              legendType="plainline"
            />
            {avgValue !== undefined && (
              <ReferenceLine
                y={avgValue}
                stroke={theme.colors.textSecondary}
                strokeDasharray="4 4"
                label={{
                  value: `AVG (${avgValue.toFixed(2)})`,
                  position: 'insideTopRight',
                  fill: theme.colors.textSecondary,
                  fontSize: 10,
                }}
              />
            )}
          </LineChart>
        </ChartPanel>
      </div>
    </ChartContainer>
  );
};
