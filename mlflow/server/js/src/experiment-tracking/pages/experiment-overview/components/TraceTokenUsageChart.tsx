import React, { useMemo, useCallback } from 'react';
import { useDesignSystemTheme, LightningIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { MetricViewType, AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
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
  timeBuckets,
}) => {
  const { theme } = useDesignSystemTheme();
  const tooltipStyle = useChartTooltipStyle();
  const xAxisProps = useChartXAxisProps();
  const legendFormatter = useChartLegendFormatter();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight(0.8, 0.2);

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

  // Create maps of tokens by timestamp using shared utility
  const sumExtractor = useCallback(
    (dp: { values?: Record<string, number> }) => dp.values?.[AggregationType.SUM] || 0,
    [],
  );
  const inputTokensMap = useTimestampValueMap(inputDataPoints, sumExtractor);
  const outputTokensMap = useTimestampValueMap(outputDataPoints, sumExtractor);

  // Prepare chart data - fill in all time buckets with 0 for missing data
  const chartData = useMemo(() => {
    return timeBuckets.map((timestampMs) => ({
      name: formatTimestampForTraceMetrics(timestampMs, timeIntervalSeconds),
      inputTokens: inputTokensMap.get(timestampMs) || 0,
      outputTokens: outputTokensMap.get(timestampMs) || 0,
    }));
  }, [timeBuckets, inputTokensMap, outputTokensMap, timeIntervalSeconds]);

  // Area colors
  const areaColors = {
    inputTokens: theme.colors.blue400,
    outputTokens: theme.colors.green400,
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
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis hide />
              <Tooltip
                contentStyle={tooltipStyle}
                cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
                formatter={(value: number, name: string) => [formatTokenCount(value), name]}
              />
              <Area
                type="monotone"
                dataKey="inputTokens"
                stackId="1"
                stroke={areaColors.inputTokens}
                fill={areaColors.inputTokens}
                strokeOpacity={getOpacity('Input Tokens')}
                fillOpacity={getOpacity('Input Tokens')}
                strokeWidth={2}
                name="Input Tokens"
              />
              <Area
                type="monotone"
                dataKey="outputTokens"
                stackId="1"
                stroke={areaColors.outputTokens}
                fill={areaColors.outputTokens}
                strokeOpacity={getOpacity('Output Tokens')}
                fillOpacity={getOpacity('Output Tokens')}
                strokeWidth={2}
                name="Output Tokens"
              />
              <Legend
                verticalAlign="bottom"
                iconType="plainline"
                height={36}
                onMouseEnter={handleLegendMouseEnter}
                onMouseLeave={handleLegendMouseLeave}
                formatter={legendFormatter}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <OverviewChartEmptyState />
        )}
      </div>
    </OverviewChartContainer>
  );
};
