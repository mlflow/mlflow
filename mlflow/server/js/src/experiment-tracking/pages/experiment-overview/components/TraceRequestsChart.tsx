import React, { useMemo } from 'react';
import { useDesignSystemTheme, ChartLineIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { MetricViewType, AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from '../hooks/useTraceMetricsQuery';
import { formatTimestampForTraceMetrics, getTimestampFromDataPoint } from '../utils/chartUtils';
import { ChartLoadingState, ChartErrorState, ChartEmptyState, ChartHeader, OverTimeLabel } from './ChartCardWrapper';
import type { OverviewChartProps } from '../types';

export const TraceRequestsChart: React.FC<OverviewChartProps> = ({
  experimentId,
  startTimeMs,
  endTimeMs,
  timeIntervalSeconds,
}) => {
  const { theme } = useDesignSystemTheme();

  // Fetch trace count metrics grouped by time bucket
  const {
    data: traceCountData,
    isLoading,
    error,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
    timeIntervalSeconds,
  });

  const traceCountDataPoints = useMemo(() => traceCountData?.data_points || [], [traceCountData?.data_points]);

  // Get total requests
  const totalRequests = useMemo(
    () => traceCountDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.COUNT] || 0), 0),
    [traceCountDataPoints],
  );

  // Prepare chart data for recharts (data is already sorted by time_bucket from backend)
  const chartData = useMemo(() => {
    return traceCountDataPoints.map((dp) => {
      const timestampMs = getTimestampFromDataPoint(dp);
      return {
        name: timestampMs ? formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds) : '',
        count: dp.values?.[AggregationType.COUNT] || 0,
      };
    });
  }, [traceCountDataPoints, timeIntervalSeconds]);

  if (isLoading) {
    return <ChartLoadingState />;
  }

  if (error) {
    return <ChartErrorState />;
  }

  return (
    <div
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        padding: theme.spacing.lg,
        backgroundColor: theme.colors.backgroundPrimary,
      }}
    >
      <ChartHeader
        icon={<ChartLineIcon />}
        title={<FormattedMessage defaultMessage="Requests" description="Title for the trace requests chart" />}
        value={totalRequests.toLocaleString()}
      />
      <OverTimeLabel />

      {/* Chart */}
      <div css={{ height: 200 }}>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <XAxis
                dataKey="name"
                tick={{ fontSize: 10, fill: theme.colors.textSecondary, dy: theme.spacing.sm }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis hide />
              <Tooltip
                contentStyle={{
                  backgroundColor: theme.colors.backgroundPrimary,
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.borders.borderRadiusMd,
                  fontSize: 12,
                }}
                cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
                formatter={(value: number) => [`${value}`, 'Requests']}
              />
              <Bar dataKey="count" fill={theme.colors.blue400} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <ChartEmptyState />
        )}
      </div>
    </div>
  );
};
