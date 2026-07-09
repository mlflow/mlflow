import React, { useCallback } from 'react';
import { LightningIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { useToolLatencyChartData } from '../hooks/useToolLatencyChartData';
import { useItemSelection } from '../hooks/useItemSelection';
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
} from './OverviewChartComponents';
import { ItemSelector } from './ItemSelector';
import { formatLatency, useLegendHighlight, useChartColors, getLineDotStyle } from '../utils/chartUtils';

/**
 * Chart showing average latency comparison for each tool over time as a line chart.
 * Each line represents a different tool with its average latency.
 */
export const ToolLatencyChart: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const xAxisProps = useChartXAxisProps();
  const yAxisProps = useChartYAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight();
  const { getChartColor } = useChartColors();

  // Fetch and process tool latency chart data
  const { chartData, toolNames, isLoading, error, hasData } = useToolLatencyChartData();

  // Tool selection state
  const { displayedItems, isAllSelected, selectorLabel, handleSelectAllToggle, handleItemToggle } = useItemSelection(
    toolNames,
    {
      allSelected: intl.formatMessage({
        defaultMessage: 'All tools',
        description: 'Label for tool selector when all tools are selected',
      }),
      noneSelected: intl.formatMessage({
        defaultMessage: 'No tools selected',
        description: 'Label for tool selector when no tools are selected',
      }),
    },
  );

  const tooltipFormatter = useCallback(
    (value: number, name: string) => [formatLatency(value), name] as [string, string],
    [],
  );

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer componentId="mlflow.charts.tool_latency">
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <OverviewChartHeader
          icon={<LightningIcon />}
          title={
            <FormattedMessage
              defaultMessage="Latency Comparison"
              description="Title for the tool latency comparison chart"
            />
          }
        />
        {hasData && (
          <ItemSelector
            componentId="mlflow.charts.tool_latency.tool_selector"
            itemNames={toolNames}
            displayedItems={displayedItems}
            isAllSelected={isAllSelected}
            selectorLabel={selectorLabel}
            onSelectAllToggle={handleSelectAllToggle}
            onItemToggle={handleItemToggle}
          />
        )}
      </div>

      {/* Chart */}
      <div css={{ height: 300, marginTop: theme.spacing.sm }}>
        {hasData && displayedItems.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
              <XAxis dataKey="timestamp" {...xAxisProps} />
              <YAxis {...yAxisProps} />
              <Tooltip
                content={
                  <ScrollableTooltip
                    formatter={tooltipFormatter}
                    componentId="mlflow.overview.usage.latency.view_traces_link"
                  />
                }
                cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
                wrapperStyle={{ pointerEvents: 'auto' }}
              />
              {displayedItems.map((toolName) => {
                const originalIndex = toolNames.indexOf(toolName);
                return (
                  <Line
                    key={toolName}
                    type="monotone"
                    dataKey={toolName}
                    stroke={getChartColor(originalIndex)}
                    strokeWidth={2}
                    strokeOpacity={getOpacity(toolName)}
                    dot={getLineDotStyle(getChartColor(originalIndex))}
                  />
                );
              })}
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
