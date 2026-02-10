import React, { useCallback, useState, useMemo } from 'react';
import {
  useDesignSystemTheme,
  ChartLineIcon,
  DialogCombobox,
  DialogComboboxTrigger,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxOptionListSearch,
  SegmentedControlGroup,
  SegmentedControlButton,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { formatCostUSD } from '@databricks/web-shared/model-trace-explorer';
import { useTraceCostOverTimeChartData } from '../hooks/useTraceCostOverTimeChartData';
import { useTraceCostDimension, type CostDimension } from '../hooks/useTraceCostDimension';
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

  // Selection state - null means all selected (default), array means specific selection
  const [selectedItems, setSelectedItems] = useState<string[] | null>(null);

  // Reset selection when dimension changes
  const [prevDimension, setPrevDimension] = useState(dimension);
  if (prevDimension !== dimension) {
    setPrevDimension(dimension);
    setSelectedItems(null);
  }

  // Compute which items to display
  const isAllSelected = selectedItems === null;
  const displayedItems = isAllSelected
    ? dimensionValues
    : selectedItems.filter((item) => dimensionValues.includes(item));

  const handleSelectAllToggle = useCallback(() => {
    setSelectedItems(isAllSelected ? [] : null);
  }, [isAllSelected]);

  const handleItemToggle = useCallback(
    (itemName: string) => {
      setSelectedItems((prev) => {
        if (prev === null) {
          return dimensionValues.filter((m) => m !== itemName);
        }
        const newSelection = prev.includes(itemName) ? prev.filter((m) => m !== itemName) : [...prev, itemName];
        return newSelection.length === dimensionValues.length ? null : newSelection;
      });
    },
    [dimensionValues],
  );

  const selectorLabel = useMemo(() => {
    const allLabel =
      dimension === 'model'
        ? intl.formatMessage({
            defaultMessage: 'All models',
            description: 'Label for selector when all models are selected',
          })
        : intl.formatMessage({
            defaultMessage: 'All providers',
            description: 'Label for selector when all providers are selected',
          });

    if (isAllSelected) {
      return allLabel;
    }
    if (displayedItems.length === 0) {
      return dimension === 'model'
        ? intl.formatMessage({
            defaultMessage: 'No models selected',
            description: 'Label for selector when no models are selected',
          })
        : intl.formatMessage({
            defaultMessage: 'No providers selected',
            description: 'Label for selector when no providers are selected',
          });
    }
    return intl.formatMessage(
      {
        defaultMessage: '{count} selected',
        description: 'Label for selector showing count of selected items',
      },
      { count: displayedItems.length },
    );
  }, [isAllSelected, displayedItems, intl, dimension]);

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
          <SegmentedControlGroup
            name="cost-over-time-dimension"
            componentId="mlflow.overview.usage.trace_cost_over_time.dimension"
            value={dimension}
            onChange={({ target: { value } }) => setDimension(value as CostDimension)}
          >
            <SegmentedControlButton value="model">
              <FormattedMessage defaultMessage="Model" description="Dimension toggle option for model" />
            </SegmentedControlButton>
            <SegmentedControlButton value="provider">
              <FormattedMessage defaultMessage="Provider" description="Dimension toggle option for provider" />
            </SegmentedControlButton>
          </SegmentedControlGroup>
          {/* Item selector dropdown */}
          {hasData && dimensionValues.length > 0 && (
            <DialogCombobox
              componentId="mlflow.overview.usage.trace_cost_over_time.item_selector"
              label={selectorLabel}
              multiSelect
              value={[]}
            >
              <DialogComboboxTrigger allowClear={false} css={{ minWidth: 120 }} data-testid="item-selector-dropdown" />
              <DialogComboboxContent maxHeight={300} align="end">
                <DialogComboboxOptionList>
                  <DialogComboboxOptionListSearch>
                    <DialogComboboxOptionListCheckboxItem
                      key="__select_all__"
                      value="__select_all__"
                      checked={isAllSelected}
                      onChange={handleSelectAllToggle}
                    >
                      <FormattedMessage
                        defaultMessage="Select All"
                        description="Option to select all items in the selector"
                      />
                    </DialogComboboxOptionListCheckboxItem>
                    {dimensionValues.map((itemName) => (
                      <DialogComboboxOptionListCheckboxItem
                        key={itemName}
                        value={itemName}
                        checked={isAllSelected || displayedItems.includes(itemName)}
                        onChange={() => handleItemToggle(itemName)}
                      >
                        {itemName}
                      </DialogComboboxOptionListCheckboxItem>
                    ))}
                  </DialogComboboxOptionListSearch>
                </DialogComboboxOptionList>
              </DialogComboboxContent>
            </DialogCombobox>
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
                content={<ScrollableTooltip formatter={tooltipFormatter} />}
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
