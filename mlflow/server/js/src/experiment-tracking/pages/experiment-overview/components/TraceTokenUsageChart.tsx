import React, { useCallback } from 'react';
import { useDesignSystemTheme, LightningIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ComposedChart, Area, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { useTraceTokenUsageChartData } from '../hooks/useTraceTokenUsageChartData';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartContainer,
  ScrollableTooltip,
  useChartXAxisProps,
  useChartYAxisProps,
  useScrollableLegendProps,
  DEFAULT_CHART_CONTENT_HEIGHT,
} from './OverviewChartComponents';
import { formatCount, useLegendHighlight, getLineDotStyle } from '../utils/chartUtils';
import { useOverviewChartContext } from '../OverviewChartContext';

export const TraceTokenUsageChart: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const xAxisProps = useChartXAxisProps();
  const yAxisProps = useChartYAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight(0.8, 0.2);
  const { experimentIds, timeIntervalSeconds } = useOverviewChartContext();

  // Fetch and process token usage chart data
  const {
    chartData,
    totalTokens,
    totalInputTokens,
    totalOutputTokens,
    totalCacheReadTokens,
    totalCacheCreationTokens,
    isLoading,
    error,
    hasData,
  } = useTraceTokenUsageChartData();

  const tooltipFormatter = useCallback(
    (value: number, name: string) => [formatCount(value), name] as [string, string],
    [],
  );

  // Area colors
  const areaColors = {
    inputTokens: theme.colors.blue400,
    outputTokens: theme.colors.green400,
    cacheReadTokens: theme.colors.yellow500,
    cacheCreationTokens: theme.colors.purple,
  };

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer componentId="mlflow.charts.trace_token_usage">
      <OverviewChartHeader
        icon={<LightningIcon />}
        title={<FormattedMessage defaultMessage="Token Usage" description="Title for the token usage chart" />}
        value={formatCount(totalTokens)}
        subtitle={(() => {
          const parts = [`${formatCount(totalInputTokens)} input`, `${formatCount(totalOutputTokens)} output`];
          if (totalCacheReadTokens > 0) parts.push(`${formatCount(totalCacheReadTokens)} cache read`);
          if (totalCacheCreationTokens > 0) parts.push(`${formatCount(totalCacheCreationTokens)} cache write`);
          return `(${parts.join(', ')})`;
        })()}
      />

      {/* Chart */}
      <div css={{ height: DEFAULT_CHART_CONTENT_HEIGHT, marginTop: theme.spacing.sm }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis {...yAxisProps} />
              <Tooltip
                content={
                  <ScrollableTooltip
                    formatter={tooltipFormatter}
                    componentId="mlflow.overview.usage.token_usage.view_traces_link"
                    linkConfig={{
                      experimentId: experimentIds[0],
                      timeIntervalSeconds,
                    }}
                  />
                }
                cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
                wrapperStyle={{ pointerEvents: 'auto' }}
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
                dot={getLineDotStyle(areaColors.inputTokens)}
                name="Input Tokens"
              />
              <Line
                type="monotone"
                dataKey="cacheReadTokens"
                stroke={areaColors.cacheReadTokens}
                strokeWidth={2}
                strokeDasharray="5 3"
                strokeOpacity={getOpacity('(Cache Read)')}
                dot={getLineDotStyle(areaColors.cacheReadTokens)}
                name="(Cache Read)"
              />
              <Line
                type="monotone"
                dataKey="cacheCreationTokens"
                stroke={areaColors.cacheCreationTokens}
                strokeWidth={2}
                strokeDasharray="5 3"
                strokeOpacity={getOpacity('(Cache Write)')}
                dot={getLineDotStyle(areaColors.cacheCreationTokens)}
                name="(Cache Write)"
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
                dot={getLineDotStyle(areaColors.outputTokens)}
                name="Output Tokens"
              />
              <Legend
                verticalAlign="bottom"
                iconType="plainline"
                onMouseEnter={handleLegendMouseEnter}
                onMouseLeave={handleLegendMouseLeave}
                {...scrollableLegendProps}
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
