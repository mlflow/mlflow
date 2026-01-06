import React, { useMemo, useCallback } from 'react';
import { useDesignSystemTheme, ClockIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  P50,
  P90,
  P99,
  getPercentileKey,
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

/**
 * Format latency value in human-readable format
 */
function formatLatency(ms: number): string {
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)} sec`;
  }
  return `${ms.toFixed(0)} ms`;
}

export const TraceLatencyChart: React.FC<OverviewChartProps> = ({
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

  // Fetch latency metrics with p50, p90, p99 aggregations grouped by time
  const {
    data: latencyData,
    isLoading: isLoadingTimeSeries,
    error: timeSeriesError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.LATENCY,
    aggregations: [
      { aggregation_type: AggregationType.PERCENTILE, percentile_value: P50 },
      { aggregation_type: AggregationType.PERCENTILE, percentile_value: P90 },
      { aggregation_type: AggregationType.PERCENTILE, percentile_value: P99 },
    ],
    timeIntervalSeconds,
  });

  // Fetch overall average latency (without time bucketing) for the header
  const {
    data: avgLatencyData,
    isLoading: isLoadingAvg,
    error: avgError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.LATENCY,
    aggregations: [{ aggregation_type: AggregationType.AVG }],
  });

  const latencyDataPoints = useMemo(() => latencyData?.data_points || [], [latencyData?.data_points]);
  const isLoading = isLoadingTimeSeries || isLoadingAvg;
  const error = timeSeriesError || avgError;

  // Extract overall average latency from the response (undefined if not available)
  const avgLatency = avgLatencyData?.data_points?.[0]?.values?.[AggregationType.AVG];

  // Create a map of latency values by timestamp
  const latencyExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => ({
      p50: dp.values?.[getPercentileKey(P50)] || 0,
      p90: dp.values?.[getPercentileKey(P90)] || 0,
      p99: dp.values?.[getPercentileKey(P99)] || 0,
    }),
    [],
  );
  const latencyByTimestamp = useTimestampValueMap(latencyDataPoints, latencyExtractor);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => {
      const latency = latencyByTimestamp.get(timestampMs);
      return {
        name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
        p50: latency?.p50 || 0,
        p90: latency?.p90 || 0,
        p99: latency?.p99 || 0,
      };
    });
  }, [timeBuckets, latencyByTimestamp, timeIntervalSeconds]);

  // Line colors
  const lineColors = {
    p50: theme.colors.blue300,
    p90: theme.colors.blue500,
    p99: theme.colors.yellow500,
  };

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer>
      <OverviewChartHeader
        icon={<ClockIcon />}
        title={<FormattedMessage defaultMessage="Latency" description="Title for the latency chart" />}
        value={avgLatency !== undefined ? formatLatency(avgLatency) : undefined}
      />

      <OverviewChartTimeLabel />

      {/* Chart */}
      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        {latencyDataPoints.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 30, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis hide />
              <Tooltip
                contentStyle={tooltipStyle}
                cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
                formatter={(value: number, name: string) => [formatLatency(value), name]}
              />
              <Line
                type="monotone"
                dataKey="p50"
                stroke={lineColors.p50}
                strokeWidth={2}
                dot={false}
                name="p50"
                strokeOpacity={getOpacity('p50')}
              />
              <Line
                type="monotone"
                dataKey="p90"
                stroke={lineColors.p90}
                strokeWidth={2}
                dot={false}
                name="p90"
                strokeOpacity={getOpacity('p90')}
              />
              <Line
                type="monotone"
                dataKey="p99"
                stroke={lineColors.p99}
                strokeWidth={2}
                dot={false}
                name="p99"
                strokeOpacity={getOpacity('p99')}
              />
              {avgLatency !== undefined && (
                <ReferenceLine
                  y={avgLatency}
                  stroke={theme.colors.textSecondary}
                  strokeDasharray="4 4"
                  label={{
                    value: `AVG (${formatLatency(avgLatency)})`,
                    position: 'insideTopRight',
                    fill: theme.colors.textSecondary,
                    fontSize: 10,
                  }}
                />
              )}
              <Legend
                verticalAlign="bottom"
                iconType="plainline"
                height={36}
                onMouseEnter={handleLegendMouseEnter}
                onMouseLeave={handleLegendMouseLeave}
                formatter={legendFormatter}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <OverviewChartEmptyState />
        )}
      </div>
    </OverviewChartContainer>
  );
};
