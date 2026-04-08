import React, { useCallback, useMemo } from 'react';
import { useDesignSystemTheme, ChartLineIcon } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { formatCostUSD } from '@databricks/web-shared/model-trace-explorer';
import { useTraceCostOverTimeChartData } from '../hooks/useTraceCostOverTimeChartData';
import { useTraceCostDimension } from '../hooks/useTraceCostDimension';
import { useItemSelection } from '../hooks/useItemSelection';
import { CostDimensionToggle } from './CostDimensionToggle';
import { ItemSelector } from './ItemSelector';
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
import { useChartColors, useLegendHighlight, getLineDotStyle } from '../utils/chartUtils';

export const TraceCostOverTimeChart: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const xAxisProps = useChartXAxisProps();
  const yAxisProps = useChartYAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { getChartColor } = useChartColors();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight(0.8, 0.2);

  const { dimension, setDimension } = useTraceCostDimension();

  // Fetch and process cost over time chart data
  const { chartData, dimensionValues, totalCost, isLoading, error, hasData } = useTraceCostOverTimeChartData(dimension);

  // Item selection labels depend on dimension
  const selectionLabels = useMemo(
    () => ({
      allSelected:
        dimension === 'model'
          ? intl.formatMessage({
              defaultMessage: 'All models',
              description: 'Label for selector when all models are selected',
            })
          : intl.formatMessage({
              defaultMessage: 'All providers',
              description: 'Label for selector when all providers are selected',
            }),
      noneSelected:
        dimension === 'model'
          ? intl.formatMessage({
              defaultMessage: 'No models selected',
              description: 'Label for selector when no models are selected',
            })
          : intl.formatMessage({
              defaultMessage: 'No providers selected',
              description: 'Label for selector when no providers are selected',
            }),
    }),
    [dimension, intl],
  );

  // Item selection state - resets when dimension changes
  const { displayedItems, isAllSelected, selectorLabel, handleSelectAllToggle, handleItemToggle } = useItemSelection(
    dimensionValues,
    selectionLabels,
    dimension,
  );

  const tooltipFormatter = useCallback(
    (value: number, name: string) => [formatCostUSD(value), name] as [string, string],
    [],
  );

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer componentId="mlflow.overview.usage.trace_cost_over_time">
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <OverviewChartHeader
          icon={<ChartLineIcon />}
          title={
            <FormattedMessage
              defaultMessage="Cost Over Time"
              description="Title for the cost over time by model chart"
            />
          }
          value={formatCostUSD(totalCost)}
          subtitle={
            <FormattedMessage defaultMessage="Total Cost" description="Subtitle for the cost over time chart total" />
          }
        />
        <div css={{ display: 'flex', gap: theme.spacing.sm, flexShrink: 0 }}>
          <CostDimensionToggle
            componentId="mlflow.overview.usage.trace_cost_over_time.dimension"
            value={dimension}
            onChange={setDimension}
          />
          {hasData && dimensionValues.length > 0 && (
            <ItemSelector
              componentId="mlflow.overview.usage.trace_cost_over_time.item_selector"
              itemNames={dimensionValues}
              data-testid="item-selector-dropdown"
              displayedItems={displayedItems}
              isAllSelected={isAllSelected}
              selectorLabel={selectorLabel}
              onSelectAllToggle={handleSelectAllToggle}
              onItemToggle={handleItemToggle}
            />
          )}
        </div>
      </div>

      {/* Chart */}
      <div css={{ height: DEFAULT_CHART_CONTENT_HEIGHT, marginTop: theme.spacing.sm }}>
        {hasData && displayedItems.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis {...yAxisProps} />
              <Tooltip
                content={
                  <ScrollableTooltip
                    formatter={tooltipFormatter}
                    componentId="mlflow.overview.usage.traces.view_traces_link"
                  />
                }
                cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
              />
              {displayedItems.map((itemName) => {
                const originalIndex = dimensionValues.indexOf(itemName);
                return (
                  <Line
                    key={itemName}
                    type="monotone"
                    dataKey={itemName}
                    stroke={getChartColor(originalIndex)}
                    strokeOpacity={getOpacity(itemName)}
                    strokeWidth={2}
                    dot={getLineDotStyle(getChartColor(originalIndex))}
                    name={itemName}
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
          <OverviewChartEmptyState
            message={
              <FormattedMessage
                defaultMessage="No cost data available"
                description="Message shown when there is no cost data to display"
              />
            }
          />
        )}
      </div>
    </OverviewChartContainer>
  );
};
