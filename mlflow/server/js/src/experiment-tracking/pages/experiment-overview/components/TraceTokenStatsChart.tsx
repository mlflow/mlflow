import React from 'react';
import { useDesignSystemTheme, BarChartIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';
import { useTraceTokenStatsChartData } from '../hooks/useTraceTokenStatsChartData';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartTimeLabel,
  OverviewChartContainer,
  useChartTooltipStyle,
  useChartXAxisProps,
  useScrollableLegendProps,
} from './OverviewChartComponents';
import { formatCount, useLegendHighlight } from '../utils/chartUtils';

export const TraceTokenStatsChart: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const tooltipStyle = useChartTooltipStyle();
  const xAxisProps = useChartXAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight();

  // Fetch and process token stats chart data
  const { chartData, avgTokens, isLoading, error, hasData } = useTraceTokenStatsChartData();

  // Line colors
  const lineColors = {
    p50: theme.colors.green300,
    p90: theme.colors.green500,
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
        icon={<BarChartIcon />}
        title={<FormattedMessage defaultMessage="Tokens per Trace" description="Title for the token stats chart" />}
        value={avgTokens !== undefined ? formatCount(Math.round(avgTokens)) : undefined}
        subtitle={
          avgTokens !== undefined ? (
            <FormattedMessage defaultMessage="avg per trace" description="Subtitle for average tokens per trace" />
          ) : undefined
        }
      />

      <OverviewChartTimeLabel />

      {/* Chart */}
      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 30, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis hide />
              <Tooltip
                contentStyle={tooltipStyle}
                cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
                formatter={(value: number, name: string) => [formatCount(value), name]}
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
                    value: `AVG (${formatCount(Math.round(avgTokens))})`,
                    position: 'insideTopRight',
                    fill: theme.colors.textSecondary,
                    fontSize: 10,
                  }}
                />
              )}
              <Legend
                verticalAlign="bottom"
                iconType="plainline"
                onMouseEnter={handleLegendMouseEnter}
                onMouseLeave={handleLegendMouseLeave}
                {...scrollableLegendProps}
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
