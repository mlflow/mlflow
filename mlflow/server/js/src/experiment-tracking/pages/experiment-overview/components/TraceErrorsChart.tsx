import React, { useMemo, useCallback } from 'react';
import { useDesignSystemTheme, DangerIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ComposedChart, Bar, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  TraceFilterKey,
  TraceStatus,
  createTraceFilter,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from '../hooks/useTraceMetricsQuery';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartContainer,
  OverviewChartTimeLabel,
  useChartTooltipStyle,
  useChartXAxisProps,
  useChartLegendFormatter,
} from './OverviewChartComponents';
import { formatTimestampForTraceMetrics, useLegendHighlight, useTimestampValueMap } from '../utils/chartUtils';
import type { OverviewChartProps } from '../types';

// Filter to get only error traces
const ERROR_FILTER = createTraceFilter(TraceFilterKey.STATUS, TraceStatus.ERROR);

export const TraceErrorsChart: React.FC<OverviewChartProps> = ({
  experimentId,
  startTimeMs,
  endTimeMs,
  timeIntervalSeconds,
  timeBuckets,
}) => {
  const { theme } = useDesignSystemTheme();
  const tooltipStyle = useChartTooltipStyle();
  const xAxisProps = useChartXAxisProps();
  const legendFormatter = useChartLegendFormatter();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight();

  // Fetch error count metrics grouped by time bucket
  const {
    data: errorCountData,
    isLoading: isLoadingErrors,
    error: errorCountError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    timeIntervalSeconds,
    filters: [ERROR_FILTER],
  });

  // Fetch total trace count metrics grouped by time bucket (for calculating error rate)
  // This query is also used by TraceRequestsChart, so React Query will dedupe it
  const {
    data: totalCountData,
    isLoading: isLoadingTotal,
    error: totalCountError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    timeIntervalSeconds,
  });

  const errorDataPoints = useMemo(() => errorCountData?.data_points || [], [errorCountData?.data_points]);
  const totalDataPoints = useMemo(() => totalCountData?.data_points || [], [totalCountData?.data_points]);
  const isLoading = isLoadingErrors || isLoadingTotal;
  const error = errorCountError || totalCountError;

  // Calculate totals by summing time-bucketed data
  const totalErrors = useMemo(
    () => errorDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.COUNT] || 0), 0),
    [errorDataPoints],
  );
  const totalTraces = useMemo(
    () => totalDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.COUNT] || 0), 0),
    [totalDataPoints],
  );
  const overallErrorRate = totalTraces > 0 ? (totalErrors / totalTraces) * 100 : 0;

  // Create maps by timestamp for easy lookup using shared utility
  const countExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => dp.values?.[AggregationType.COUNT] || 0,
    [],
  );
  const errorCountByTimestamp = useTimestampValueMap(errorDataPoints, countExtractor);
  const totalCountByTimestamp = useTimestampValueMap(totalDataPoints, countExtractor);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => {
      const errorCount = errorCountByTimestamp.get(timestampMs) || 0;
      const totalCount = totalCountByTimestamp.get(timestampMs) || 0;
      const errorRate = totalCount > 0 ? (errorCount / totalCount) * 100 : 0;

      return {
        name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
        errorCount,
        errorRate: Math.round(errorRate * 100) / 100, // Round to 2 decimal places
      };
    });
  }, [timeBuckets, errorCountByTimestamp, totalCountByTimestamp, timeIntervalSeconds]);

  // Calculate average error rate across time buckets for the reference line
  const avgErrorRate = useMemo(() => {
    if (chartData.length === 0) return 0;
    const sum = chartData.reduce((acc, dp) => acc + dp.errorRate, 0);
    return sum / chartData.length;
  }, [chartData]);

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer>
      <OverviewChartHeader
        icon={<DangerIcon />}
        title={<FormattedMessage defaultMessage="Errors" description="Title for the errors chart" />}
        value={totalErrors.toLocaleString()}
        subtitle={`(Overall error rate: ${overallErrorRate.toFixed(1)}%)`}
      />

      <OverviewChartTimeLabel />

      {/* Chart */}
      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        {totalDataPoints.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis yAxisId="left" hide />
              <YAxis yAxisId="right" domain={[0, 100]} hide />
              <Tooltip
                contentStyle={tooltipStyle}
                cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
                formatter={(value: number, name: string) => {
                  if (name === 'Error Count') {
                    return [value.toLocaleString(), name];
                  }
                  return [`${value.toFixed(1)}%`, name];
                }}
              />
              <Bar
                yAxisId="left"
                dataKey="errorCount"
                fill={theme.colors.red400}
                radius={[4, 4, 0, 0]}
                name="Error Count"
                fillOpacity={getOpacity('Error Count')}
                legendType="square"
              />
              {avgErrorRate > 0 && (
                <ReferenceLine
                  yAxisId="right"
                  y={avgErrorRate}
                  stroke={theme.colors.textSecondary}
                  strokeDasharray="4 4"
                  label={{
                    value: `AVG (${avgErrorRate.toFixed(1)}%)`,
                    position: 'insideTopRight',
                    fill: theme.colors.textSecondary,
                    fontSize: 10,
                  }}
                />
              )}
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="errorRate"
                stroke={theme.colors.yellow500}
                strokeWidth={2}
                dot={false}
                name="Error Rate"
                strokeOpacity={getOpacity('Error Rate')}
                legendType="plainline"
              />
              <Legend
                verticalAlign="bottom"
                height={36}
                onMouseEnter={handleLegendMouseEnter}
                onMouseLeave={handleLegendMouseLeave}
                formatter={legendFormatter}
              />
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <OverviewChartEmptyState />
        )}
      </div>
    </OverviewChartContainer>
  );
};
