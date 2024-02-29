import {
  Button,
  Empty,
  Input,
  RefreshIcon,
  SearchIcon,
  Spacer,
  Spinner,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { compact, mapValues, values } from 'lodash';
import { useMemo, useState } from 'react';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { FormattedMessage, defineMessages, useIntl } from 'react-intl';
import { useSelector } from 'react-redux';
import { getGridColumnSetup } from '../../../common/utils/CssGrid.utils';
import { ReduxState } from '../../../redux-types';
import { RunInfoEntity } from '../../types';
import { normalizeChartMetricKey } from '../../utils/MetricsUtils';
import { useChartMoveUpDownFunctions, useOrderedCharts } from '../runs-charts/hooks/useOrderedCharts';
import { RunsChartsTooltipWrapper } from '../runs-charts/hooks/useRunsChartsTooltip';
import { RunViewChartTooltipBody } from './RunViewChartTooltipBody';
import { RunViewMetricChart } from './RunViewMetricChart';
import { ChartRefreshManager, useChartRefreshManager } from './useChartRefreshManager';

const { systemMetricChartsEmpty, modelMetricChartsEmpty } = defineMessages({
  systemMetricChartsEmpty: {
    defaultMessage: 'No system metrics recorded',
    description: 'Run page > Charts tab > System charts section > empty label',
  },
  modelMetricChartsEmpty: {
    defaultMessage: 'No model metrics recorded',
    description: 'Run page > Charts tab > Model charts section > empty label',
  },
});

const EmptyMetricsFiltered = () => (
  <Empty
    title="No matching metric keys"
    description="All metrics in this section are filtered. Clear the search filter to see hidden metrics."
  />
);

const EmptyMetricsNotRecorded = ({ label }: { label: React.ReactNode }) => <Empty title={label} description={null} />;

const metricKeyMatchesFilter = (filter: string, metricKey: string) =>
  metricKey.toLowerCase().startsWith(filter.toLowerCase()) ||
  normalizeChartMetricKey(metricKey).toLowerCase().startsWith(filter.toLowerCase());

/**
 * Internal component that displays a single collapsible section with charts
 */
const RunViewMetricChartsSection = ({
  metricKeys,
  search,
  runInfo,
  chartRefreshManager,
  onReorderChart,
}: {
  metricKeys: string[];
  search: string;
  runInfo: RunInfoEntity;
  onReorderChart: (sourceChartKey: string, targetChartKey: string) => void;
  chartRefreshManager: ChartRefreshManager;
}) => {
  const { theme } = useDesignSystemTheme();

  const filteredMetricKeys = metricKeys.filter((metricKey) => metricKeyMatchesFilter(search, metricKey));

  const { canMoveDown, canMoveUp, moveChartDown, moveChartUp } = useChartMoveUpDownFunctions(
    filteredMetricKeys,
    onReorderChart,
  );

  const gridSetup = useMemo(
    () => ({
      ...getGridColumnSetup({
        maxColumns: 3,
        gap: theme.spacing.lg,
        additionalBreakpoints: [{ breakpointWidth: 3 * 720, minColumnWidthForBreakpoint: 720 }],
      }),
      overflow: 'hidden',
    }),
    [theme],
  );

  return filteredMetricKeys.length ? (
    <div css={gridSetup}>
      {filteredMetricKeys.map((metricKey, index) => (
        <RunViewMetricChart
          // Use both metric name and index as a key,
          // charts needs to be rerendered when order is changed
          key={`${metricKey}-${index}`}
          dragGroupKey="metricCharts"
          metricKey={metricKey}
          runInfo={runInfo}
          onReorderWith={onReorderChart}
          canMoveUp={canMoveUp(metricKey)}
          canMoveDown={canMoveDown(metricKey)}
          onMoveDown={() => moveChartDown(metricKey)}
          onMoveUp={() => moveChartUp(metricKey)}
          chartRefreshManager={chartRefreshManager}
        />
      ))}
    </div>
  ) : (
    <EmptyMetricsFiltered />
  );
};

/**
 * Component displaying metric charts for a single run
 */
export const RunViewMetricCharts = ({
  runInfo,
  metricKeys,
  mode,
}: {
  metricKeys: string[];
  runInfo: RunInfoEntity;
  /**
   * Whether to display model or system metrics. This affects labels and tooltips.
   */
  mode: 'model' | 'system';
}) => {
  const chartRefreshManager = useChartRefreshManager();

  const metricsForRun = useSelector(({ entities }: ReduxState) => {
    return mapValues(entities.sampledMetricsByRunUuid[runInfo.run_uuid], (metricsByRange) => {
      return compact(
        values(metricsByRange)
          .map(({ metricsHistory }) => metricsHistory)
          .flat(),
      );
    });
  });

  const [search, setSearch] = useState('');
  const { formatMessage } = useIntl();

  const { orderedMetricKeys, onReorderChart } = useOrderedCharts(metricKeys, 'RunView' + mode, runInfo.run_uuid);

  const noMetricsRecorded = !metricKeys.length;
  const allMetricsFilteredOut =
    !noMetricsRecorded && !metricKeys.some((metricKey) => metricKeyMatchesFilter(search, metricKey));
  const showConfigArea = !noMetricsRecorded;
  const { theme } = useDesignSystemTheme();
  const showCharts = !noMetricsRecorded && !allMetricsFilteredOut;

  const tooltipContext = useMemo(() => ({ runInfo, metricsForRun }), [runInfo, metricsForRun]);

  const anyRunRefreshing = useSelector((store: ReduxState) => {
    return values(store.entities.sampledMetricsByRunUuid[runInfo.run_uuid]).some((metricsByRange) =>
      values(metricsByRange).some(({ refreshing }) => refreshing),
    );
  });

  return (
    <DndProvider backend={HTML5Backend}>
      <div css={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <div css={{ flexShrink: 0 }}>
          {showConfigArea && (
            <div css={{ display: 'flex', gap: theme.spacing.sm }}>
              <Input
                role="searchbox"
                prefix={<SearchIcon />}
                value={search}
                allowClear
                onChange={(e) => setSearch(e.target.value)}
                placeholder={formatMessage({
                  defaultMessage: 'Search metric charts',
                  description: 'Run page > Charts tab > Filter metric charts input > placeholder',
                })}
              />
              <Button
                componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewmetriccharts.tsx_176"
                icon={
                  anyRunRefreshing ? <Spinner size="small" css={{ marginRight: theme.spacing.sm }} /> : <RefreshIcon />
                }
                onClick={() => {
                  if (!anyRunRefreshing) {
                    chartRefreshManager.refreshAllCharts();
                  }
                }}
              >
                <FormattedMessage
                  defaultMessage="Refresh"
                  description="Run page > Charts tab > Refresh all charts button label"
                />
              </Button>
            </div>
          )}
          <Spacer />
        </div>
        {noMetricsRecorded && (
          <EmptyMetricsNotRecorded
            label={<FormattedMessage {...(mode === 'model' ? modelMetricChartsEmpty : systemMetricChartsEmpty)} />}
          />
        )}
        {allMetricsFilteredOut && <EmptyMetricsFiltered />}
        {/* Scroll charts independently from filter box */}
        <div css={{ flex: 1, overflow: 'auto' }}>
          <RunsChartsTooltipWrapper contextData={tooltipContext} component={RunViewChartTooltipBody} hoverOnly>
            {showCharts && (
              <RunViewMetricChartsSection
                metricKeys={orderedMetricKeys}
                runInfo={runInfo}
                search={search}
                onReorderChart={onReorderChart}
                chartRefreshManager={chartRefreshManager}
              />
            )}
          </RunsChartsTooltipWrapper>
        </div>{' '}
      </div>
    </DndProvider>
  );
};
