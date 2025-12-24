import React, { useMemo, useState } from 'react';
import { useDesignSystemTheme, LightningIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import {
  MetricViewType,
  AggregationType,
  TraceMetricKey,
  TIME_BUCKET_DIMENSION_KEY,
} from '@databricks/web-shared/model-trace-explorer';
import { useTraceMetricsQuery } from '../hooks/useTraceMetricsQuery';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartTimeLabel,
} from './OverviewChartComponents';
import { formatTimestampForTraceMetrics, generateTimeBuckets } from '../utils/chartUtils';
import type { OverviewChartProps } from '../types';

/**
 * Format token count in human-readable format (K, M)
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

export const TraceTokenUsageChart: React.FC<OverviewChartProps> = ({
  experimentId,
  startTimeMs,
  endTimeMs,
  timeIntervalSeconds,
}) => {
  const { theme } = useDesignSystemTheme();

  // Fetch input tokens over time
  const {
    data: inputTokensData,
    isLoading: isLoadingInput,
    error: inputError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.INPUT_TOKENS,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
    timeIntervalSeconds,
  });

  // Fetch output tokens over time
  const {
    data: outputTokensData,
    isLoading: isLoadingOutput,
    error: outputError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.OUTPUT_TOKENS,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
    timeIntervalSeconds,
  });

  // Fetch total tokens (without time bucketing) for the header
  const {
    data: totalTokensData,
    isLoading: isLoadingTotal,
    error: totalError,
  } = useTraceMetricsQuery({
    experimentId,
    startTimeMs,
    endTimeMs,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TOTAL_TOKENS,
    aggregations: [{ aggregation_type: AggregationType.SUM }],
  });

  const inputDataPoints = useMemo(() => inputTokensData?.data_points || [], [inputTokensData?.data_points]);
  const outputDataPoints = useMemo(() => outputTokensData?.data_points || [], [outputTokensData?.data_points]);
  const isLoading = isLoadingInput || isLoadingOutput || isLoadingTotal;
  const error = inputError || outputError || totalError;

  // Extract total tokens from the response
  const totalTokens = totalTokensData?.data_points?.[0]?.values?.[AggregationType.SUM] || 0;

  // Calculate total input and output tokens from time-bucketed data
  const totalInputTokens = useMemo(
    () => inputDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.SUM] || 0), 0),
    [inputDataPoints],
  );
  const totalOutputTokens = useMemo(
    () => outputDataPoints.reduce((sum, dp) => sum + (dp.values?.[AggregationType.SUM] || 0), 0),
    [outputDataPoints],
  );

  // Create maps of tokens by timestamp for merging
  const inputTokensMap = useMemo(() => {
    const map = new Map<number, number>();
    for (const dp of inputDataPoints) {
      const timeBucket = dp.dimensions?.[TIME_BUCKET_DIMENSION_KEY];
      if (timeBucket) {
        const ts = new Date(timeBucket).getTime();
        map.set(ts, dp.values?.[AggregationType.SUM] || 0);
      }
    }
    return map;
  }, [inputDataPoints]);

  const outputTokensMap = useMemo(() => {
    const map = new Map<number, number>();
    for (const dp of outputDataPoints) {
      const timeBucket = dp.dimensions?.[TIME_BUCKET_DIMENSION_KEY];
      if (timeBucket) {
        const ts = new Date(timeBucket).getTime();
        map.set(ts, dp.values?.[AggregationType.SUM] || 0);
      }
    }
    return map;
  }, [outputDataPoints]);

  // Generate all time buckets within the selected range
  const allTimeBuckets = useMemo(
    () => generateTimeBuckets(startTimeMs, endTimeMs, timeIntervalSeconds),
    [startTimeMs, endTimeMs, timeIntervalSeconds],
  );

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return allTimeBuckets.map((timestampMs) => ({
      name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
      inputTokens: inputTokensMap.get(timestampMs) || 0,
      outputTokens: outputTokensMap.get(timestampMs) || 0,
    }));
  }, [allTimeBuckets, inputTokensMap, outputTokensMap, timeIntervalSeconds]);

  // Track hovered legend item
  const [hoveredArea, setHoveredArea] = useState<string | null>(null);

  // Area colors
  const areaColors = {
    inputTokens: theme.colors.blue400,
    outputTokens: theme.colors.green400,
  };

  // Get opacity for an area based on hover state
  const getAreaOpacity = (areaKey: string) => {
    if (hoveredArea === null) return 0.8;
    return hoveredArea === areaKey ? 0.8 : 0.2;
  };

  // Handle legend hover
  const handleLegendMouseEnter = (data: { value: string }) => {
    setHoveredArea(data.value);
  };

  const handleLegendMouseLeave = () => {
    setHoveredArea(null);
  };

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
        icon={<LightningIcon />}
        title={<FormattedMessage defaultMessage="Token Usage" description="Title for the token usage chart" />}
        value={formatTokenCount(totalTokens)}
        subtitle={`(${formatTokenCount(totalInputTokens)} input, ${formatTokenCount(totalOutputTokens)} output)`}
      />

      <OverviewChartTimeLabel />

      {/* Chart */}
      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        {inputDataPoints.length > 0 || outputDataPoints.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 30, bottom: 0 }}>
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
                  fontSize: theme.typography.fontSizeSm,
                }}
                cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
                formatter={(value: number, name: string) => [formatTokenCount(value), name]}
              />
              <Area
                type="monotone"
                dataKey="inputTokens"
                stackId="1"
                stroke={areaColors.inputTokens}
                fill={areaColors.inputTokens}
                strokeOpacity={getAreaOpacity('Input Tokens')}
                fillOpacity={getAreaOpacity('Input Tokens')}
                strokeWidth={2}
                name="Input Tokens"
              />
              <Area
                type="monotone"
                dataKey="outputTokens"
                stackId="1"
                stroke={areaColors.outputTokens}
                fill={areaColors.outputTokens}
                strokeOpacity={getAreaOpacity('Output Tokens')}
                fillOpacity={getAreaOpacity('Output Tokens')}
                strokeWidth={2}
                name="Output Tokens"
              />
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
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <OverviewChartEmptyState />
        )}
      </div>
    </div>
  );
};
