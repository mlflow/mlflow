import React, { useMemo, useCallback } from 'react';
import { useDesignSystemTheme } from '@databricks/design-system';
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
  ChartLoadingState,
  ChartErrorState,
  ChartEmptyState,
  ChartHeader,
  OverTimeLabel,
  ChartContainer,
  useChartTooltipStyle,
  useChartXAxisProps,
  useChartLegendFormatter,
} from './ChartCardWrapper';
import { formatTimestampForTraceMetrics, useLegendHighlight, useTimestampValueMap } from '../utils/chartUtils';
import type { OverviewChartProps } from '../types';

// Icon component for token stats (bar chart style)
const TokenStatsIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
    <path d="M2 12h2V6H2v6zm4 0h2V4H6v8zm4 0h2V2h-2v10zm4 0h2V8h-2v4z" />
  </svg>
);

/**
 * Format token count in human-readable format
 */
function formatTokenCount(count: number): string {
  if (count >= 1_000_000) {
    return `${(count / 1_000_000).toFixed(2)}M`;
  }
  if (count >= 1_000) {
    return `${(count / 1_000).toFixed(2)}K`;
  }
  return count.toLocaleString();
}

export const TraceTokenStatsChart: React.FC<OverviewChartProps> = ({
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

  // Fetch token stats with p50, p90, p99 aggregations grouped by time
  const {
    data: tokenStatsData,
    isLoading: isLoadingTimeSeries,
    error: timeSeriesError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TOTAL_TOKENS,
    aggregations: [
      { aggregation_type: AggregationType.PERCENTILE, percentile_value: P50 },
      { aggregation_type: AggregationType.PERCENTILE, percentile_value: P90 },
      { aggregation_type: AggregationType.PERCENTILE, percentile_value: P99 },
    ],
    timeIntervalSeconds,
  });

  // Fetch overall average tokens (without time bucketing) for the header
  const {
    data: avgTokensData,
    isLoading: isLoadingAvg,
    error: avgError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TOTAL_TOKENS,
    aggregations: [{ aggregation_type: AggregationType.AVG }],
  });

  const tokenStatsDataPoints = useMemo(() => tokenStatsData?.data_points || [], [tokenStatsData?.data_points]);
  const isLoading = isLoadingTimeSeries || isLoadingAvg;
  const error = timeSeriesError || avgError;

  // Extract overall average tokens from the response (undefined if not available)
  const avgTokens = avgTokensData?.data_points?.[0]?.values?.[AggregationType.AVG];

  // Create a map of token stats by timestamp using shared utility
  const statsExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => ({
      p50: dp.values?.[getPercentileKey(P50)] || 0,
      p90: dp.values?.[getPercentileKey(P90)] || 0,
      p99: dp.values?.[getPercentileKey(P99)] || 0,
    }),
    [],
  );
  const tokenStatsByTimestamp = useTimestampValueMap(tokenStatsDataPoints, statsExtractor);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => {
      const stats = tokenStatsByTimestamp.get(timestampMs);
      return {
        name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
        p50: stats?.p50 || 0,
        p90: stats?.p90 || 0,
        p99: stats?.p99 || 0,
      };
    });
  }, [timeBuckets, tokenStatsByTimestamp, timeIntervalSeconds]);

  // Line colors
  const lineColors = {
    p50: theme.colors.green300,
    p90: theme.colors.green500,
    p99: theme.colors.yellow500,
  };

  if (isLoading) {
    return <ChartLoadingState />;
  }

  if (error) {
    return <ChartErrorState />;
  }

  return (
    <ChartContainer>
      <ChartHeader
        icon={<TokenStatsIcon />}
        title={<FormattedMessage defaultMessage="Tokens per Trace" description="Title for the token stats chart" />}
        value={avgTokens !== undefined ? formatTokenCount(Math.round(avgTokens)) : undefined}
        subtitle={
          avgTokens !== undefined ? (
            <FormattedMessage defaultMessage="avg per trace" description="Subtitle for average tokens per trace" />
          ) : undefined
        }
      />

      <OverTimeLabel />

      {/* Chart */}
      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        {tokenStatsDataPoints.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 30, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis hide />
              <Tooltip
                contentStyle={tooltipStyle}
                cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
                formatter={(value: number, name: string) => [formatTokenCount(value), name]}
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
              {avgTokens !== undefined && (
                <ReferenceLine
                  y={avgTokens}
                  stroke={theme.colors.textSecondary}
                  strokeDasharray="4 4"
                  label={{
                    value: `AVG (${formatTokenCount(Math.round(avgTokens))})`,
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
          <ChartEmptyState />
        )}
      </div>
    </ChartContainer>
  );
};
