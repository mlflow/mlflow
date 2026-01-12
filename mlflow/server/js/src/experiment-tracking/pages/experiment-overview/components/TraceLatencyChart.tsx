import React from 'react';
import { useDesignSystemTheme, ClockIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';
import { useTraceLatencyChartData } from '../hooks/useTraceLatencyChartData';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartContainer,
  OverviewChartTimeLabel,
  useChartTooltipStyle,
  useChartXAxisProps,
  useScrollableLegendProps,
} from './OverviewChartComponents';
import { useLegendHighlight } from '../utils/chartUtils';

/**
 * Format latency value in human-readable format
 */
function formatLatency(ms: number): string {
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)} sec`;
  }
  return `${ms.toFixed(0)} ms`;
}

export const TraceLatencyChart: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const tooltipStyle = useChartTooltipStyle();
  const xAxisProps = useChartXAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight();

  // Fetch and process latency chart data
  const { chartData, avgLatency, isLoading, error, hasData } = useTraceLatencyChartData();

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
        {hasData ? (
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
