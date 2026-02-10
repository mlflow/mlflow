import React, { useCallback } from 'react';
import { useDesignSystemTheme, DangerIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ComposedChart, Bar, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts';
import { useNavigate } from '../../../../common/utils/RoutingUtils';
import { useTraceErrorsChartData } from '../hooks/useTraceErrorsChartData';
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
  getTracesFilteredByTimeRangeUrl,
  createSpanStatusEqualsFilter,
} from './OverviewChartComponents';
import { useLegendHighlight, getLineDotStyle } from '../utils/chartUtils';
import { useOverviewChartContext } from '../OverviewChartContext';

export const TraceErrorsChart: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const xAxisProps = useChartXAxisProps();
  const yAxisProps = useChartYAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight();
  const { experimentId, timeIntervalSeconds } = useOverviewChartContext();
  const navigate = useNavigate();

  // Fetch and process errors chart data
  const { chartData, totalErrors, overallErrorRate, avgErrorRate, isLoading, error, hasData } =
    useTraceErrorsChartData();

  const tooltipFormatter = useCallback((value: number, name: string) => {
    if (name === 'Error Count') {
      return [value.toLocaleString(), name] as [string, string];
    }
    return [`${value.toFixed(1)}%`, name] as [string, string];
  }, []);

  // Handle click on tooltip link to navigate to traces filtered by error status
  const handleViewTraces = useCallback(
    (_label: string | undefined, dataPoint?: { timestampMs?: number }) => {
      if (dataPoint?.timestampMs === undefined) return;
      const url = getTracesFilteredByTimeRangeUrl(experimentId, dataPoint.timestampMs, timeIntervalSeconds, [
        createSpanStatusEqualsFilter('ERROR'),
      ]);
      navigate(url);
    },
    [experimentId, timeIntervalSeconds, navigate],
  );

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer componentId="mlflow.charts.trace_errors">
      <OverviewChartHeader
        icon={<DangerIcon />}
        title={<FormattedMessage defaultMessage="Errors" description="Title for the errors chart" />}
        value={totalErrors.toLocaleString()}
        subtitle={`(Overall error rate: ${overallErrorRate.toFixed(1)}%)`}
      />

      {/* Chart */}
      <div css={{ height: DEFAULT_CHART_CONTENT_HEIGHT, marginTop: theme.spacing.sm }}>
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis yAxisId="left" {...yAxisProps} />
              <YAxis
                yAxisId="right"
                orientation="right"
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
                {...yAxisProps}
              />
              <Tooltip
                content={
                  <ScrollableTooltip
                    formatter={tooltipFormatter}
                    linkConfig={{
                      componentId: 'mlflow.overview.usage.errors.view_traces_link',
                      onLinkClick: handleViewTraces,
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
                dot={getLineDotStyle(theme.colors.yellow500)}
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
