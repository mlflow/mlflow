import React, { useMemo } from 'react';
import { useDesignSystemTheme, ChartLineIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  TIME_BUCKET_DIMENSION_KEY,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from '../hooks/useTraceMetricsQuery';
import { formatTimestampForTraceMetrics, generateTimeBuckets } from '../utils/chartUtils';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartTimeLabel,
} from './OverviewChartComponents';
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

  // Create a map of counts by timestamp
  const countByTimestamp = useMemo(() => {
    const map = new Map<number, number>();
    for (const dp of traceCountDataPoints) {
      const timeBucket = dp.dimensions?.[TIME_BUCKET_DIMENSION_KEY];
      if (timeBucket) {
        const ts = new Date(timeBucket).getTime();
        map.set(ts, dp.values?.[AggregationType.COUNT] || 0);
      }
    }
    return map;
  }, [traceCountDataPoints]);

  // Generate all time buckets within the selected range
  const allTimeBuckets = useMemo(
    () => generateTimeBuckets(startTimeMs, endTimeMs, timeIntervalSeconds),
    [startTimeMs, endTimeMs, timeIntervalSeconds],
  );

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return allTimeBuckets.map((timestampMs) => ({
      name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
      count: countByTimestamp.get(timestampMs) || 0,
    }));
  }, [allTimeBuckets, countByTimestamp, timeIntervalSeconds]);

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
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
      <OverviewChartHeader
        icon={<ChartLineIcon />}
        title={<FormattedMessage defaultMessage="Requests" description="Title for the trace requests chart" />}
        value={totalRequests.toLocaleString()}
      />
      <OverviewChartTimeLabel />

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
          <OverviewChartEmptyState />
        )}
      </div>
    </div>
  );
};
