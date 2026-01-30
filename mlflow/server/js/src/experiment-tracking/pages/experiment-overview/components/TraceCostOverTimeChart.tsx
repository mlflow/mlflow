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
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { formatCostUSD } from '@databricks/web-shared/model-trace-explorer';
import { useTraceCostOverTimeChartData } from '../hooks/useTraceCostOverTimeChartData';
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

const SELECT_ALL_VALUE = '__SELECT_ALL__';

export const TraceCostOverTimeChart: React.FC = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const xAxisProps = useChartXAxisProps();
  const yAxisProps = useChartYAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { getChartColor } = useChartColors();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight(0.8, 0.2);

  // Fetch and process cost over time chart data
  const { chartData, modelNames, totalCost, isLoading, error, hasData } = useTraceCostOverTimeChartData();

  // Model selection state - null means all models selected (default), array means specific selection
  const [selectedModels, setSelectedModels] = useState<string[] | null>(null);

  // Compute which models to display
  const isAllSelected = selectedModels === null;
  const displayedModels = isAllSelected ? modelNames : selectedModels.filter((model) => modelNames.includes(model));

  const handleModelToggle = useCallback(
    (modelName: string) => {
      if (modelName === SELECT_ALL_VALUE) {
        // Toggle: if all selected -> select none, if not all -> select all
        setSelectedModels(isAllSelected ? [] : null);
      } else {
        setSelectedModels((prev) => {
          if (prev === null) {
            // Switching from "all" to specific - deselect this one model
            return modelNames.filter((m) => m !== modelName);
          }
          const newSelection = prev.includes(modelName) ? prev.filter((m) => m !== modelName) : [...prev, modelName];
          // If all models are now selected, switch back to "all" state
          return newSelection.length === modelNames.length ? null : newSelection;
        });
      }
    },
    [isAllSelected, modelNames],
  );

  const modelSelectorLabel = useMemo(() => {
    if (isAllSelected) {
      return intl.formatMessage({
        defaultMessage: 'All models',
        description: 'Label for model selector when all models are selected',
      });
    }
    if (displayedModels.length === 0) {
      return intl.formatMessage({
        defaultMessage: 'No models selected',
        description: 'Label for model selector when no models are selected',
      });
    }
    return intl.formatMessage(
      {
        defaultMessage: '{count} selected',
        description: 'Label for model selector showing count of selected models',
      },
      { count: displayedModels.length },
    );
  }, [isAllSelected, displayedModels, intl]);

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
        {/* Model selector dropdown */}
        {hasData && modelNames.length > 0 && (
          <div css={{ flexShrink: 0 }}>
            <DialogCombobox
              componentId="mlflow.overview.usage.trace_cost_over_time.model_selector"
              label={modelSelectorLabel}
              multiSelect
              value={[]}
            >
              <DialogComboboxTrigger allowClear={false} css={{ minWidth: 120 }} data-testid="model-selector-dropdown" />
              <DialogComboboxContent maxHeight={300} align="end">
                <DialogComboboxOptionList>
                  <DialogComboboxOptionListSearch>
                    <DialogComboboxOptionListCheckboxItem
                      key={SELECT_ALL_VALUE}
                      value={SELECT_ALL_VALUE}
                      checked={isAllSelected}
                      onChange={() => handleModelToggle(SELECT_ALL_VALUE)}
                    >
                      <FormattedMessage
                        defaultMessage="Select All"
                        description="Option to select all models in the model selector"
                      />
                    </DialogComboboxOptionListCheckboxItem>
                    {modelNames.map((modelName) => (
                      <DialogComboboxOptionListCheckboxItem
                        key={modelName}
                        value={modelName}
                        checked={isAllSelected || displayedModels.includes(modelName)}
                        onChange={() => handleModelToggle(modelName)}
                      >
                        {modelName}
                      </DialogComboboxOptionListCheckboxItem>
                    ))}
                  </DialogComboboxOptionListSearch>
                </DialogComboboxOptionList>
              </DialogComboboxContent>
            </DialogCombobox>
          </div>
        )}
      </div>

      {/* Chart */}
      <div css={{ height: DEFAULT_CHART_CONTENT_HEIGHT, marginTop: theme.spacing.sm }}>
        {hasData && displayedModels.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
              <XAxis dataKey="name" {...xAxisProps} />
              <YAxis {...yAxisProps} />
              <Tooltip
                content={<ScrollableTooltip formatter={tooltipFormatter} />}
                cursor={{ stroke: theme.colors.actionTertiaryBackgroundHover }}
              />
              {displayedModels.map((modelName) => {
                const originalIndex = modelNames.indexOf(modelName);
                return (
                  <Line
                    key={modelName}
                    type="monotone"
                    dataKey={modelName}
                    stroke={getChartColor(originalIndex)}
                    strokeOpacity={getOpacity(modelName)}
                    strokeWidth={2}
                    dot={getLineDotStyle(getChartColor(originalIndex))}
                    name={modelName}
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
