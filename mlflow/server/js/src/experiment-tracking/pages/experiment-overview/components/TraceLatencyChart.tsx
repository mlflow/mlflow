import React, { useMemo, useState } from 'react';
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
import { ChartLoadingState, ChartErrorState, ChartEmptyState, ChartHeader, OverTimeLabel } from './ChartCardWrapper';
import { formatTimestampForTraceMetrics, getTimestampFromDataPoint } from '../utils/chartUtils';
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
}) => {
  const { theme } = useDesignSystemTheme();

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

  // Prepare chart data for recharts
  const chartData = useMemo(() => {
    return latencyDataPoints.map((dp) => {
      const timestampMs = getTimestampFromDataPoint(dp);
      return {
        name: timestampMs ? formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds) : '',
        p50: dp.values?.[getPercentileKey(P50)] || 0,
        p90: dp.values?.[getPercentileKey(P90)] || 0,
        p99: dp.values?.[getPercentileKey(P99)] || 0,
      };
    });
  }, [latencyDataPoints, timeIntervalSeconds]);

  // Track hovered legend item (null means none hovered, show all)
  const [hoveredLine, setHoveredLine] = useState<string | null>(null);

  // Line colors
  const lineColors = {
    p50: theme.colors.blue300,
    p90: theme.colors.blue500,
    p99: theme.colors.yellow500,
  };

  // Get opacity for a line based on hover state
  const getLineOpacity = (lineKey: string) => {
    if (hoveredLine === null) return 1;
    return hoveredLine === lineKey ? 1 : 0.2;
  };

  // Handle legend hover
  const handleLegendMouseEnter = (data: { value: string }) => {
    setHoveredLine(data.value);
  };

  const handleLegendMouseLeave = () => {
    setHoveredLine(null);
  };

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
        icon={<ClockIcon />}
        title={<FormattedMessage defaultMessage="Latency" description="Title for the latency chart" />}
        value={avgLatency !== undefined ? formatLatency(avgLatency) : undefined}
      />

      <OverTimeLabel />

      {/* Chart */}
      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 30, bottom: 0 }}>
              <XAxis
                dataKey="name"
                tick={{ fontSize: 10, fill: theme.colors.textSecondary, dy: theme.spacing.sm }}
                axisLine={false}
                tickLine={false}
                interval="preserveStartEnd"
              />
              <YAxis hide />
              <Tooltip
                contentStyle={{
                  backgroundColor: theme.colors.backgroundPrimary,
                  border: `1px solid ${theme.colors.border}`,
                  borderRadius: theme.borders.borderRadiusMd,
                  fontSize: 12,
                }}
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
                strokeOpacity={getLineOpacity('p50')}
              />
              <Line
                type="monotone"
                dataKey="p90"
                stroke={lineColors.p90}
                strokeWidth={2}
                dot={false}
                name="p90"
                strokeOpacity={getLineOpacity('p90')}
              />
              <Line
                type="monotone"
                dataKey="p99"
                stroke={lineColors.p99}
                strokeWidth={2}
                dot={false}
                name="p99"
                strokeOpacity={getLineOpacity('p99')}
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
                formatter={(value) => (
                  <span
                    style={{
                      color: theme.colors.textPrimary,
                      fontSize: theme.typography.fontSizeSm,
                      cursor: 'pointer',
                    }}
                  >
                    {value}
                  </span>
                )}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <ChartEmptyState />
        )}
      </div>
    </div>
  );
};
