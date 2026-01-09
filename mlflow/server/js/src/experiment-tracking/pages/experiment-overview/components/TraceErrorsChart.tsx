import React from 'react';
import { useDesignSystemTheme, DangerIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ComposedChart, Bar, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';
import { useTraceErrorsChartData } from '../hooks/useTraceErrorsChartData';
import {
  OverviewChartLoadingState,
  OverviewChartErrorState,
  OverviewChartEmptyState,
  OverviewChartHeader,
  OverviewChartContainer,
  OverviewChartTimeLabel,
  ScrollableTooltip,
  useChartXAxisProps,
  useScrollableLegendProps,
} from './OverviewChartComponents';
import { useLegendHighlight } from '../utils/chartUtils';

export const TraceErrorsChart: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const xAxisProps = useChartXAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight();

  // Fetch and process errors chart data
  const { chartData, totalErrors, overallErrorRate, avgErrorRate, isLoading, error, hasData } =
    useTraceErrorsChartData();

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer>
      <OverviewChartHeader
        icon={<DangerIcon />}
        title={<FormattedMessage defaultMessage="Errors" description="Title for the errors chart" />}
        value={totalErrors.toLocaleString()}
        subtitle={`(Overall error rate: ${overallErrorRate.toFixed(1)}%)`}
      />

      <OverviewChartTimeLabel />

      {/* Chart */}
      <div css={{ height: 200, marginTop: theme.spacing.sm }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis yAxisId="left" hide />
              <YAxis yAxisId="right" domain={[0, 100]} hide />
              <Tooltip
                content={
                  <ScrollableTooltip
                    formatter={(value, name) => {
                      if (name === 'Error Count') {
                        return [value.toLocaleString(), name];
                      }
                      return [`${value.toFixed(1)}%`, name];
                    }}
                  />
                }
                cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
              />
              <Bar
                yAxisId="left"
                dataKey="errorCount"
                fill={theme.colors.red400}
                radius={[4, 4, 0, 0]}
                name="Error Count"
                fillOpacity={getOpacity('Error Count')}
                legendType="square"
              />
              {avgErrorRate > 0 && (
                <ReferenceLine
                  yAxisId="right"
                  y={avgErrorRate}
                  stroke={theme.colors.textSecondary}
                  strokeDasharray="4 4"
                  label={{
                    value: `AVG (${avgErrorRate.toFixed(1)}%)`,
                    position: 'insideTopRight',
                    fill: theme.colors.textSecondary,
                    fontSize: 10,
                  }}
                />
              )}
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="errorRate"
                stroke={theme.colors.yellow500}
                strokeWidth={2}
                dot={false}
                name="Error Rate"
                strokeOpacity={getOpacity('Error Rate')}
                legendType="plainline"
              />
              <Legend
                verticalAlign="bottom"
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
