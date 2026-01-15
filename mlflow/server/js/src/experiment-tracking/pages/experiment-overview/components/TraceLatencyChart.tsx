import React, { useCallback } from 'react';
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
  ScrollableTooltip,
  useChartXAxisProps,
  useChartYAxisProps,
  useScrollableLegendProps,
  DEFAULT_CHART_CONTENT_HEIGHT,
} from './OverviewChartComponents';
import { formatLatency, useLegendHighlight } from '../utils/chartUtils';

export const TraceLatencyChart: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const xAxisProps = useChartXAxisProps();
  const yAxisProps = useChartYAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight();

  // Fetch and process latency chart data
  const { chartData, avgLatency, isLoading, error, hasData } = useTraceLatencyChartData();

  const tooltipFormatter = useCallback(
    (value: number, name: string) => [formatLatency(value), name] as [string, string],
    [],
  );

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
    <OverviewChartContainer componentId="mlflow.charts.trace_latency">
      <OverviewChartHeader
        icon={<ClockIcon />}
        title={<FormattedMessage defaultMessage="Latency" description="Title for the latency chart" />}
        value={avgLatency !== undefined ? formatLatency(avgLatency) : undefined}
      />

      <OverviewChartTimeLabel />

      {/* Chart */}
      <div css={{ height: DEFAULT_CHART_CONTENT_HEIGHT, marginTop: theme.spacing.sm }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis {...yAxisProps} />
              <Tooltip
                content={<ScrollableTooltip formatter={tooltipFormatter} />}
                cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
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
