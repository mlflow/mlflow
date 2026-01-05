import React, { useMemo, useCallback } from 'react';
import { useDesignSystemTheme, ChartLineIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { MetricViewType, AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from '../hooks/useTraceMetricsQuery';
import { formatTimestampForTraceMetrics, useTimestampValueMap } from '../utils/chartUtils';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartTimeLabel,
  OverviewChartContainer,
  useChartTooltipStyle,
  useChartXAxisProps,
} from './OverviewChartComponents';
import type { OverviewChartProps } from '../types';

export const TraceRequestsChart: React.FC<OverviewChartProps> = ({
  experimentId,
  startTimeMs,
  endTimeMs,
  timeIntervalSeconds,
  timeBuckets,
}) => {
  const { theme } = useDesignSystemTheme();
  const tooltipStyle = useChartTooltipStyle();
  const xAxisProps = useChartXAxisProps();

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

  // Create a map of counts by timestamp using shared utility
  const countExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => dp.values?.[AggregationType.COUNT] || 0,
    [],
  );
  const countByTimestamp = useTimestampValueMap(traceCountDataPoints, countExtractor);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => ({
      name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
      count: countByTimestamp.get(timestampMs) || 0,
    }));
  }, [timeBuckets, countByTimestamp, timeIntervalSeconds]);

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer>
      <OverviewChartHeader
        icon={<ChartLineIcon />}
        title={<FormattedMessage defaultMessage="Requests" description="Title for the trace requests chart" />}
        value={totalRequests.toLocaleString()}
      />
      <OverviewChartTimeLabel />

      {/* Chart */}
      <div css={{ height: 200 }}>
        {traceCountDataPoints.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis hide />
              <Tooltip
                contentStyle={tooltipStyle}
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
    </OverviewChartContainer>
  );
};
