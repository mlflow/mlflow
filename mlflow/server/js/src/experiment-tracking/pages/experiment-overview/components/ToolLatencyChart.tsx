import React, { useCallback } from 'react';
import { LightningIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { useToolLatencyChartData } from '../hooks/useToolLatencyChartData';
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
  useClickableTooltip,
  LockedTooltipOverlay,
} from './OverviewChartComponents';
import { formatLatency, useLegendHighlight, useChartColors, getLineDotStyle } from '../utils/chartUtils';

/**
 * Chart showing average latency comparison for each tool over time as a line chart.
 * Each line represents a different tool with its average latency.
 */
export const ToolLatencyChart: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const xAxisProps = useChartXAxisProps();
  const yAxisProps = useChartYAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight();
  const { getChartColor } = useChartColors();

  const { lockedTooltip, isLocked, containerRef, tooltipRef, handleChartClick } = useClickableTooltip();

  // Fetch and process tool latency chart data
  const { chartData, toolNames, isLoading, error, hasData } = useToolLatencyChartData();

  const tooltipFormatter = useCallback(
    (value: number, name: string) => [formatLatency(value), name] as [string, string],
    [],
  );

  const onChartClick = useCallback(
    (data: any, event: React.MouseEvent) => handleChartClick(data, event),
    [handleChartClick],
  );

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer componentId="mlflow.charts.tool_latency">
      <OverviewChartHeader
        icon={<LightningIcon />}
        title={
          <FormattedMessage
            defaultMessage="Latency Comparison"
            description="Title for the tool latency comparison chart"
          />
        }
      />

      {/* Chart */}
      <div ref={containerRef} css={{ height: DEFAULT_CHART_CONTENT_HEIGHT, marginTop: theme.spacing.sm, position: 'relative' }}>
        {hasData ? (
          <>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }} onClick={onChartClick}>
                <XAxis dataKey="timestamp" {...xAxisProps} />
                <YAxis {...yAxisProps} />
                <Tooltip
                  content={isLocked ? () => null : <ScrollableTooltip formatter={tooltipFormatter} />}
                  cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
                />
                {toolNames.map((toolName, index) => (
                  <Line
                    key={toolName}
                    type="monotone"
                    dataKey={toolName}
                    stroke={getChartColor(index)}
                    strokeWidth={2}
                    strokeOpacity={getOpacity(toolName)}
                    dot={getLineDotStyle(getChartColor(index))}
                    cursor="pointer"
                  />
                ))}
                <Legend
                  verticalAlign="bottom"
                  iconType="plainline"
                  onMouseEnter={handleLegendMouseEnter}
                  onMouseLeave={handleLegendMouseLeave}
                  {...scrollableLegendProps}
                />
              </LineChart>
            </ResponsiveContainer>
            {lockedTooltip && (
              <LockedTooltipOverlay data={lockedTooltip} tooltipRef={tooltipRef} formatter={tooltipFormatter} />
            )}
          </>
        ) : (
          <OverviewChartEmptyState />
        )}
      </div>
    </OverviewChartContainer>
  );
};
