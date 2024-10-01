import { Input, SearchIcon, ToggleButton, useDesignSystemTheme } from '@databricks/design-system';
import { compact, mapValues, values } from 'lodash';
import { ReactNode, useEffect, useMemo, useState } from 'react';
import { useIntl } from 'react-intl';
import { useSelector } from 'react-redux';
import { ReduxState } from '../../../redux-types';
import type { KeyValueEntity, MetricEntitiesByName, RunInfoEntity } from '../../types';

import { RunsChartsTooltipWrapper } from '../runs-charts/hooks/useRunsChartsTooltip';
import { RunViewChartTooltipBody } from './RunViewChartTooltipBody';
import { RunsChartType, RunsChartsCardConfig } from '../runs-charts/runs-charts.types';
import type { RunsChartsRunData } from '../runs-charts/components/RunsCharts.common';
import { RunsChartsLineChartXAxisType } from '../runs-charts/components/RunsCharts.common';
import type { ExperimentRunsChartsUIConfiguration } from '../experiment-page/models/ExperimentPageUIState';
import { RunsChartsSectionAccordion } from '../runs-charts/components/sections/RunsChartsSectionAccordion';
import { RunsChartsConfigureModal } from '../runs-charts/components/RunsChartsConfigureModal';
import {
  RunsChartsUIConfigurationContextProvider,
  useConfirmChartCardConfigurationFn,
  useInsertRunsChartsFn,
  useRemoveRunsChartFn,
  useReorderRunsChartsFn,
} from '../runs-charts/hooks/useRunsChartsUIConfiguration';
import { MLFLOW_MODEL_METRIC_NAME, MLFLOW_SYSTEM_METRIC_NAME, MLFLOW_SYSTEM_METRIC_PREFIX } from '../../constants';
import LocalStorageUtils from '../../../common/utils/LocalStorageUtils';
import { RunsChartsFullScreenModal } from '../runs-charts/components/RunsChartsFullScreenModal';
import { useIsTabActive } from '../../../common/hooks/useIsTabActive';
import {
  shouldEnableGlobalLineChartConfig,
  shouldEnableDraggableChartsGridLayout,
  shouldEnableImageGridCharts,
  shouldEnableRunDetailsPageAutoRefresh,
  shouldUseRegexpBasedChartFiltering,
} from '../../../common/utils/FeatureUtils';
import { usePopulateImagesByRunUuid } from '../experiment-page/hooks/usePopulateImagesByRunUuid';
import { DragAndDropProvider } from '../../../common/hooks/useDragAndDropElement';
import type { UseGetRunQueryResponseRunInfo } from './hooks/useGetRunQuery';
import { RunsChartsGlobalChartSettingsDropdown } from '../runs-charts/components/RunsChartsGlobalChartSettingsDropdown';
import { RunsChartsDraggableCardsGridContextProvider } from '../runs-charts/components/RunsChartsDraggableCardsGridContext';
import { RunsChartsFilterInput } from '../runs-charts/components/RunsChartsFilterInput';

interface RunViewMetricChartsV2Props {
  metricKeys: string[];
  runInfo: RunInfoEntity | UseGetRunQueryResponseRunInfo;
  /**
   * Whether to display model or system metrics. This affects labels and tooltips.
   */
  mode: 'model' | 'system';

  latestMetrics?: MetricEntitiesByName;
  tags?: Record<string, KeyValueEntity>;
  params?: Record<string, KeyValueEntity>;
}

/**
 * Component displaying metric charts for a single run
 */
export const RunViewMetricChartsV2Impl = ({
  runInfo,
  metricKeys,
  mode,
  chartUIState,
  updateChartsUIState,
  latestMetrics = {},
  params = {},
  tags = {},
}: RunViewMetricChartsV2Props & {
  chartUIState: ExperimentRunsChartsUIConfiguration;
  updateChartsUIState: (
    stateSetter: (state: ExperimentRunsChartsUIConfiguration) => ExperimentRunsChartsUIConfiguration,
  ) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const [search, setSearch] = useState('');
  const { formatMessage } = useIntl();

  const { compareRunCharts, compareRunSections, chartsSearchFilter } = chartUIState;

  // For the draggable grid layout, we filter visible cards on this level
  const visibleChartCards = useMemo(() => {
    if (!shouldEnableDraggableChartsGridLayout()) {
      return compareRunCharts;
    }
    return compareRunCharts?.filter((chart) => !chart.deleted) ?? [];
  }, [compareRunCharts]);

  const [fullScreenChart, setFullScreenChart] = useState<
    { config: RunsChartsCardConfig; title: string; subtitle: ReactNode } | undefined
  >(undefined);

  const metricsForRun = useSelector(({ entities }: ReduxState) => {
    return mapValues(entities.sampledMetricsByRunUuid[runInfo.runUuid ?? ''], (metricsByRange) => {
      return compact(
        values(metricsByRange)
          .map(({ metricsHistory }) => metricsHistory)
          .flat(),
      );
    });
  });

  const tooltipContextValue = useMemo(() => ({ runInfo, metricsForRun }), [runInfo, metricsForRun]);

  const { imagesByRunUuid } = useSelector((state: ReduxState) => ({
    imagesByRunUuid: state.entities.imagesByRunUuid,
  }));

  const [configuredCardConfig, setConfiguredCardConfig] = useState<RunsChartsCardConfig | null>(null);

  const reorderCharts = useReorderRunsChartsFn();

  const addNewChartCard = (metricSectionId: string) => (type: RunsChartType) =>
    setConfiguredCardConfig(RunsChartsCardConfig.getEmptyChartCardByType(type, false, undefined, metricSectionId));

  const insertCharts = useInsertRunsChartsFn();

  const startEditChart = (chartCard: RunsChartsCardConfig) => setConfiguredCardConfig(chartCard);

  const removeChart = useRemoveRunsChartFn();

  const confirmChartCardConfiguration = useConfirmChartCardConfigurationFn();

  const submitForm = (configuredCard: Partial<RunsChartsCardConfig>) => {
    confirmChartCardConfiguration(configuredCard);

    // Hide the modal
    setConfiguredCardConfig(null);
  };

  // Create a single run data object to be used in charts
  const chartData: RunsChartsRunData[] = useMemo(
    () => [
      {
        displayName: runInfo.runName ?? '',
        metrics: latestMetrics,
        params,
        tags,
        images: imagesByRunUuid[runInfo.runUuid ?? ''] || {},
        metricHistory: {},
        uuid: runInfo.runUuid ?? '',
        color: theme.colors.primary,
        runInfo,
      },
    ],
    [runInfo, latestMetrics, params, tags, imagesByRunUuid, theme],
  );

  useEffect(() => {
    if ((!compareRunSections || !compareRunCharts) && chartData.length > 0) {
      const { resultChartSet, resultSectionSet } = RunsChartsCardConfig.getBaseChartAndSectionConfigs({
        runsData: chartData,
        enabledSectionNames: [mode === 'model' ? MLFLOW_MODEL_METRIC_NAME : MLFLOW_SYSTEM_METRIC_NAME],
        // Filter only model or system metrics
        filterMetricNames: (name) => {
          const isSystemMetric = name.startsWith(MLFLOW_SYSTEM_METRIC_PREFIX);
          return mode === 'model' ? !isSystemMetric : isSystemMetric;
        },
      });

      updateChartsUIState((current) => ({
        ...current,
        compareRunCharts: resultChartSet,
        compareRunSections: resultSectionSet,
      }));
    }
  }, [compareRunCharts, compareRunSections, chartData, mode, updateChartsUIState]);

  /**
   * Update charts with the latest metrics if new are found
   */
  useEffect(() => {
    updateChartsUIState((current) => {
      if (!current.compareRunCharts || !current.compareRunSections) {
        return current;
      }
      const { resultChartSet, resultSectionSet, isResultUpdated } = RunsChartsCardConfig.updateChartAndSectionConfigs({
        compareRunCharts: current.compareRunCharts,
        compareRunSections: current.compareRunSections,
        runsData: chartData,
        isAccordionReordered: current.isAccordionReordered,
        // Filter only model or system metrics
        filterMetricNames: (name) => {
          const isSystemMetric = name.startsWith(MLFLOW_SYSTEM_METRIC_PREFIX);
          return mode === 'model' ? !isSystemMetric : isSystemMetric;
        },
      });

      if (!isResultUpdated) {
        return current;
      }
      return {
        ...current,
        compareRunCharts: resultChartSet,
        compareRunSections: resultSectionSet,
      };
    });
  }, [chartData, updateChartsUIState, mode]);

  const isTabActive = useIsTabActive();
  const autoRefreshEnabled = chartUIState.autoRefreshEnabled && shouldEnableRunDetailsPageAutoRefresh() && isTabActive;

  usePopulateImagesByRunUuid({
    runUuids: [runInfo.runUuid ?? ''],
    runUuidsIsActive: [runInfo.status === 'RUNNING'],
    autoRefreshEnabled,
    enabled: shouldEnableImageGridCharts(),
  });

  const searchChartsValue = shouldUseRegexpBasedChartFiltering() ? chartsSearchFilter ?? '' : search;

  return (
    <div
      css={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      <div
        css={{
          paddingBottom: theme.spacing.md,
          display: 'flex',
          gap: theme.spacing.sm,
          flex: '0 0 auto',
        }}
      >
        {shouldUseRegexpBasedChartFiltering() ? (
          <RunsChartsFilterInput chartsSearchFilter={chartsSearchFilter} />
        ) : (
          <Input
            componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewmetricchartsv2.tsx_230"
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
        )}
        {shouldEnableRunDetailsPageAutoRefresh() && (
          <ToggleButton
            componentId="codegen_mlflow_app_src_experiment-tracking_components_run-page_runviewmetricchartsv2.tsx_244"
            pressed={chartUIState.autoRefreshEnabled}
            onPressedChange={(pressed) => {
              updateChartsUIState((current) => ({ ...current, autoRefreshEnabled: pressed }));
            }}
          >
            {formatMessage({
              defaultMessage: 'Auto-refresh',
              description: 'Run page > Charts tab > Auto-refresh toggle button',
            })}
          </ToggleButton>
        )}
        {shouldEnableGlobalLineChartConfig() && (
          <RunsChartsGlobalChartSettingsDropdown
            metricKeyList={metricKeys}
            globalLineChartConfig={chartUIState.globalLineChartConfig}
            updateUIState={updateChartsUIState}
          />
        )}
      </div>
      <div
        css={{
          flex: 1,
          overflow: 'auto',
        }}
      >
        <RunsChartsTooltipWrapper contextData={tooltipContextValue} component={RunViewChartTooltipBody}>
          <RunsChartsDraggableCardsGridContextProvider visibleChartCards={visibleChartCards}>
            <RunsChartsSectionAccordion
              compareRunSections={compareRunSections}
              compareRunCharts={visibleChartCards}
              reorderCharts={reorderCharts}
              insertCharts={insertCharts}
              chartData={chartData}
              startEditChart={startEditChart}
              removeChart={removeChart}
              addNewChartCard={addNewChartCard}
              search={searchChartsValue}
              supportedChartTypes={[RunsChartType.LINE, RunsChartType.BAR, RunsChartType.IMAGE]}
              setFullScreenChart={setFullScreenChart}
              autoRefreshEnabled={autoRefreshEnabled}
              globalLineChartConfig={chartUIState.globalLineChartConfig}
              groupBy={null}
            />
          </RunsChartsDraggableCardsGridContextProvider>
        </RunsChartsTooltipWrapper>
      </div>
      {configuredCardConfig && (
        <RunsChartsConfigureModal
          chartRunData={chartData}
          metricKeyList={metricKeys}
          paramKeyList={[]}
          config={configuredCardConfig}
          onSubmit={submitForm}
          onCancel={() => setConfiguredCardConfig(null)}
          groupBy={null}
          supportedChartTypes={[RunsChartType.LINE, RunsChartType.BAR, RunsChartType.IMAGE]}
          globalLineChartConfig={chartUIState.globalLineChartConfig}
        />
      )}
      <RunsChartsFullScreenModal
        fullScreenChart={fullScreenChart}
        onCancel={() => setFullScreenChart(undefined)}
        chartData={chartData}
        tooltipContextValue={tooltipContextValue}
        tooltipComponent={RunViewChartTooltipBody}
        autoRefreshEnabled={autoRefreshEnabled}
        groupBy={null}
      />
    </div>
  );
};

export const RunViewMetricChartsV2 = (props: RunViewMetricChartsV2Props) => {
  const persistenceIdentifier = `${props.runInfo.runUuid}-${props.mode}`;

  const localStore = useMemo(
    () => LocalStorageUtils.getStoreForComponent('RunPage', persistenceIdentifier),
    [persistenceIdentifier],
  );

  const [chartUIState, updateChartsUIState] = useState<ExperimentRunsChartsUIConfiguration>(() => {
    const defaultChartState: ExperimentRunsChartsUIConfiguration = {
      isAccordionReordered: false,
      compareRunCharts: undefined,
      compareRunSections: undefined,
      // Auto-refresh is enabled by default only if the flag is set
      autoRefreshEnabled: shouldEnableRunDetailsPageAutoRefresh(),
      globalLineChartConfig: {
        xAxisKey: RunsChartsLineChartXAxisType.STEP,
        lineSmoothness: 0,
        selectedXAxisMetricKey: '',
      },
    };
    try {
      const persistedChartState = localStore.getItem('chartUIState');

      if (!persistedChartState) {
        return defaultChartState;
      }
      return JSON.parse(persistedChartState);
    } catch {
      return defaultChartState;
    }
  });

  useEffect(() => {
    localStore.setItem('chartUIState', JSON.stringify(chartUIState));
  }, [chartUIState, localStore]);

  return (
    <RunsChartsUIConfigurationContextProvider updateChartsUIState={updateChartsUIState}>
      <RunViewMetricChartsV2Impl {...props} chartUIState={chartUIState} updateChartsUIState={updateChartsUIState} />
    </RunsChartsUIConfigurationContextProvider>
  );
};
