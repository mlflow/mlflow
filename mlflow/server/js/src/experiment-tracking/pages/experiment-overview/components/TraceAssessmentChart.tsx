import React, { useMemo, useCallback } from 'react';
import { CheckCircleIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import {
  MetricViewType,
  AggregationType,
  AssessmentMetricKey,
  AssessmentFilterKey,
  AssessmentType,
  createAssessmentFilter,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from '../hooks/useTraceMetricsQuery';
import {
  ChartLoadingState,
  ChartErrorState,
  ChartEmptyState,
  ChartHeader,
  OverTimeLabel,
  ChartContainer,
  useChartTooltipStyle,
  useChartXAxisProps,
} from './ChartCardWrapper';
import { formatTimestampForTraceMetrics, useTimestampValueMap } from '../utils/chartUtils';
import type { OverviewChartProps } from '../types';

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

  // Use provided color or default to green
  const chartLineColor = lineColor || theme.colors.green500;

  // Create filters for feedback assessments with the given name
  const filters = useMemo(() => [createAssessmentFilter(AssessmentFilterKey.NAME, assessmentName)], [assessmentName]);

  // Fetch assessment values over time
  const {
    data: timeSeriesData,
    isLoading,
    error,
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

  const timeSeriesDataPoints = useMemo(() => timeSeriesData?.data_points || [], [timeSeriesData?.data_points]);

  // Create a map of values by timestamp
  const valueExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => dp.values?.[AggregationType.AVG] || 0,
    [],
  );
  const valuesByTimestamp = useTimestampValueMap(timeSeriesDataPoints, valueExtractor);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => {
      const value = valuesByTimestamp.get(timestampMs);
      return {
        name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
        value: value || 0,
      };
    });
  }, [timeBuckets, valuesByTimestamp, timeIntervalSeconds]);

  if (isLoading) {
    return <ChartLoadingState />;
  }

  if (error) {
    return <ChartErrorState />;
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

      <OverTimeLabel />

      {/* Chart */}
      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        {timeSeriesDataPoints.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 30, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis hide />
              <Tooltip
                contentStyle={tooltipStyle}
                cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
                formatter={(value: number) => [value.toFixed(2), assessmentName]}
              />
              <Line type="monotone" dataKey="value" stroke={chartLineColor} strokeWidth={2} dot={false} />
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
          </ResponsiveContainer>
        ) : (
          <ChartEmptyState />
        )}
      </div>
    </ChartContainer>
  );
};
