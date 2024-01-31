import {
  Button,
  Empty,
  Input,
  RefreshIcon,
  SearchIcon,
  Spacer,
  Spinner,
  useDesignSystemTheme,
  Accordion,
} from '@databricks/design-system';
import { compact, mapValues, values } from 'lodash';
import { useMemo, useState } from 'react';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { FormattedMessage, defineMessages, useIntl } from 'react-intl';
import { useSelector } from 'react-redux';
import { ReduxState } from '../../../redux-types';
import { RunInfoEntity, RunViewMetricConfig } from '../../types';
import { normalizeChartMetricKey } from '../../utils/MetricsUtils';
import { useChartMoveUpDownFunctionsV2, useOrderedChartsV2 } from '../runs-charts/hooks/useOrderedChartsV2';
import { RunsChartsTooltipWrapper } from '../runs-charts/hooks/useRunsChartsTooltip';
import { RunViewChartTooltipBody } from './RunViewChartTooltipBody';
import { ChartRefreshManager, useChartRefreshManager } from './useChartRefreshManager';
import {
  MLFLOW_MODEL_METRIC_NAME,
  MLFLOW_MODEL_METRIC_PREFIX,
  MLFLOW_SYSTEM_METRIC_NAME,
  MLFLOW_SYSTEM_METRIC_PREFIX,
} from 'experiment-tracking/constants';
import MetricChartsAccordion from '../MetricChartsAccordion';
import { RunViewMetricChartsSection } from './RunViewMetricChartsSection';
import { RunsCompareCardConfig } from '../runs-compare/runs-compare.types';

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
    title={
      <FormattedMessage
        defaultMessage="No matching metric keys"
        description="Run page > Charts tab > No matching metric keys"
      />
    }
    description={
      <FormattedMessage
        defaultMessage="All metrics in this section are filtered. Clear the search filter to see hidden metrics."
        description="Run page > Charts tab > No matching metric keys > description"
      />
    }
  />
);

const EmptyMetricsNotRecorded = ({ label }: { label: React.ReactNode }) => <Empty title={label} description={null} />;

const metricKeyMatchesFilter = (filter: string, metricKey: string) =>
  metricKey.toLowerCase().startsWith(filter.toLowerCase()) ||
  normalizeChartMetricKey(metricKey).toLowerCase().startsWith(filter.toLowerCase());

/**
 * Internal component that displays a single page with sections of charts
 */
const RunViewMetricChartsPage = ({
  metricConfigs,
  search,
  runInfo,
  chartRefreshManager,
  onReorderChart,
  onInsertChart,
}: {
  metricConfigs: RunViewMetricConfig[];
  search: string;
  runInfo: RunInfoEntity;
  onReorderChart: (sourceChartKey: string, targetChartKey: string) => void;
  onInsertChart: (sourceChartKey: string, targetSectionKey: string) => void;
  chartRefreshManager: ChartRefreshManager;
}) => {
  const filteredMetricConfigs = metricConfigs.filter((metricConfig) =>
    metricKeyMatchesFilter(search, metricConfig.metricKey),
  );

  const { canMoveDown, canMoveUp, moveChartDown, moveChartUp } = useChartMoveUpDownFunctionsV2(
    filteredMetricConfigs,
    onReorderChart,
  );

  const sections = useMemo(() => {
    const sectionSet = new Set<string>();
    filteredMetricConfigs.forEach((metricConfig: RunViewMetricConfig) => {
      if (search === '') {
        sectionSet.add(RunsCompareCardConfig.extractChartSectionName(metricConfig.metricKey)); // Extract section key from metricKey
      } else {
        sectionSet.add(metricConfig.sectionKey);
      }
    });
    return Array.from(sectionSet).sort();
  }, [filteredMetricConfigs, search]);

  // Move model metrics to the end
  const modelMetricsIndex = sections.findIndex((section) => section === MLFLOW_MODEL_METRIC_NAME);
  if (modelMetricsIndex !== -1) {
    sections.splice(modelMetricsIndex, 1);
    sections.push(MLFLOW_MODEL_METRIC_NAME);
  }

  return filteredMetricConfigs.length ? (
    <MetricChartsAccordion>
      {sections.map((sectionKey) => {
        const groupMetricConfigs = filteredMetricConfigs.filter(
          (metricConfig) => metricConfig.sectionKey === sectionKey,
        );
        return (
          <Accordion.Panel header={sectionKey + ` (${groupMetricConfigs.length})`} key={sectionKey}>
            <RunViewMetricChartsSection
              section={sectionKey}
              metricConfigs={groupMetricConfigs}
              onInsertChart={onInsertChart}
              onReorderChart={onReorderChart}
              canMoveUp={canMoveUp}
              canMoveDown={canMoveDown}
              moveChartDown={moveChartDown}
              moveChartUp={moveChartUp}
              runInfo={runInfo}
              chartRefreshManager={chartRefreshManager}
            />
          </Accordion.Panel>
        );
      })}
    </MetricChartsAccordion>
  ) : (
    <EmptyMetricsFiltered />
  );
};

/**
 * Component displaying metric charts for a single run
 */
export const RunViewMetricChartsV2 = ({
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

  // Add group infomation to metricKey
  const metricConfigs = metricKeys.map((key: string) => {
    const section = RunsCompareCardConfig.extractChartSectionName(key);
    return { metricKey: key, sectionKey: section };
  });

  const { orderedMetricConfigs, onReorderChart, onInsertChart } = useOrderedChartsV2(
    metricConfigs,
    'RunView' + mode,
    runInfo.run_uuid,
  );

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
              <RunViewMetricChartsPage
                metricConfigs={orderedMetricConfigs}
                runInfo={runInfo}
                search={search}
                onReorderChart={onReorderChart}
                onInsertChart={onInsertChart}
                chartRefreshManager={chartRefreshManager}
              />
            )}
          </RunsChartsTooltipWrapper>
        </div>{' '}
      </div>
    </DndProvider>
  );
};
