import { useCallback, useMemo } from 'react';
import { SparkleDoubleIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import {
  OverviewChartContainer,
  OverviewChartEmptyState,
  OverviewChartErrorState,
  OverviewChartHeader,
  OverviewChartLoadingState,
  ScrollableTooltip,
  useChartXAxisProps,
  useChartYAxisProps,
  useScrollableLegendProps,
} from '../../experiment-tracking/pages/experiment-overview/components/OverviewChartComponents';
import { ItemSelector } from '../../experiment-tracking/pages/experiment-overview/components/ItemSelector';
import { useItemSelection } from '../../experiment-tracking/pages/experiment-overview/hooks/useItemSelection';
import { useOverviewChartContext } from '../../experiment-tracking/pages/experiment-overview/OverviewChartContext';
import { formatCount, useLegendHighlight } from '../../experiment-tracking/pages/experiment-overview/utils/chartUtils';
import {
  useGatewayFeatureRequestsChartData,
  type GatewayFeatureRequestsDataPoint,
} from '../hooks/useGatewayFeatureRequestsChartData';
import { GatewayFeatureUsageLabels, useGatewayFeatureUsageColor } from '../utils/gatewayFeatures';

/**
 * Gateway requests over time as a stacked bar chart, with a bar segment for each built-in MLflow
 * feature and one for "Other". The selector hides features, so several can be compared together.
 */
export const GatewayFeatureRequestsChart = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const xAxisProps = useChartXAxisProps();
  const yAxisProps = useChartYAxisProps();
  const scrollableLegendProps = useScrollableLegendProps();
  const { getOpacity, handleLegendMouseEnter, handleLegendMouseLeave } = useLegendHighlight();
  const getFeatureColor = useGatewayFeatureUsageColor();
  const { experimentIds, timeIntervalSeconds } = useOverviewChartContext();
  const { chartData, series, hasData, isLoading, error } = useGatewayFeatureRequestsChartData();

  // The selector and the legend show feature labels, so selection is by label.
  const seriesLabels = useMemo(
    () => series.map((key) => intl.formatMessage(GatewayFeatureUsageLabels[key])),
    [series, intl],
  );
  const { displayedItems, isAllSelected, selectorLabel, handleSelectAllToggle, handleItemToggle } = useItemSelection(
    seriesLabels,
    {
      allSelected: intl.formatMessage({
        defaultMessage: 'All features',
        description: 'Gateway usage > requests by feature chart > selector label when all features are shown',
      }),
      noneSelected: intl.formatMessage({
        defaultMessage: 'No features selected',
        description: 'Gateway usage > requests by feature chart > selector label when no features are shown',
      }),
    },
  );

  const tooltipFormatter = useCallback(
    (value: number, name: string) => [formatCount(value), name] as [string, string],
    [],
  );

  if (isLoading) {
    return <OverviewChartLoadingState />;
  }

  if (error) {
    return <OverviewChartErrorState />;
  }

  return (
    <OverviewChartContainer componentId="mlflow.gateway.usage.feature_requests">
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <OverviewChartHeader
          icon={<SparkleDoubleIcon />}
          title={
            <FormattedMessage
              defaultMessage="Requests by Feature"
              description="Title for the gateway chart of requests over time by built-in MLflow feature"
            />
          }
        />
        {hasData && (
          <ItemSelector
            componentId="mlflow.gateway.usage.feature_requests.feature_selector"
            data-testid="gateway-feature-selector"
            itemNames={seriesLabels}
            displayedItems={displayedItems}
            isAllSelected={isAllSelected}
            selectorLabel={selectorLabel}
            onSelectAllToggle={handleSelectAllToggle}
            onItemToggle={handleItemToggle}
          />
        )}
      </div>

      <div css={{ height: 300, marginTop: theme.spacing.sm }}>
        {hasData && displayedItems.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
              <XAxis dataKey="timestamp" {...xAxisProps} />
              <YAxis {...yAxisProps} />
              <Tooltip
                content={
                  <ScrollableTooltip
                    formatter={tooltipFormatter}
                    componentId="mlflow.overview.usage.traces.view_traces_link"
                    linkConfig={{ experimentId: experimentIds[0], timeIntervalSeconds }}
                  />
                }
                cursor={{ fill: theme.colors.actionTertiaryBackgroundHover }}
                wrapperStyle={{ pointerEvents: 'auto' }}
              />
              {series.map((key, index) => {
                const label = seriesLabels[index];
                if (!displayedItems.includes(label)) {
                  return null;
                }
                return (
                  <Bar
                    key={key}
                    name={label}
                    dataKey={(dataPoint: GatewayFeatureRequestsDataPoint) => dataPoint.values[key]}
                    stackId="features"
                    fill={getFeatureColor(key)}
                    fillOpacity={getOpacity(label)}
                  />
                );
              })}
              {/* Keeps the series order instead of sorting by name, so "Other" stays last as in the table. */}
              <Legend
                verticalAlign="bottom"
                iconType="square"
                itemSorter={null}
                onMouseEnter={handleLegendMouseEnter}
                onMouseLeave={handleLegendMouseLeave}
                {...scrollableLegendProps}
              />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <OverviewChartEmptyState />
        )}
      </div>
    </OverviewChartContainer>
  );
};
