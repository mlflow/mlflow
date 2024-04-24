import { LegacySkeleton, useDesignSystemTheme } from '@databricks/design-system';
import type { Theme } from '@emotion/react';
import { ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { connect, useSelector } from 'react-redux';
import type {
  ExperimentStoreEntities,
  KeyValueEntity,
  MetricEntitiesByName,
  MetricHistoryByName,
  ChartSectionConfig,
  UpdateExperimentSearchFacetsFn,
  RunInfoEntity,
} from '../../types';
import { RunsChartsCardConfig } from '../runs-charts/runs-charts.types';
import type { RunsChartType, SerializedRunsChartsCardConfigCard } from '../runs-charts/runs-charts.types';
import { RunsChartsConfigureModal } from '../runs-charts/components/RunsChartsConfigureModal';
import { getUUID } from '../../../common/utils/ActionUtils';
import type { RunsChartsRunData } from '../runs-charts/components/RunsCharts.common';
import { AUTOML_EVALUATION_METRIC_TAG, MLFLOW_EXPERIMENT_PRIMARY_METRIC_NAME } from '../../constants';
import { RunsChartsTooltipBody } from '../runs-charts/components/RunsChartsTooltipBody';
import { RunsChartsTooltipWrapper } from '../runs-charts/hooks/useRunsChartsTooltip';
import { useMultipleChartsMetricHistory } from './hooks/useMultipleChartsMetricHistory';
import { useUpdateExperimentViewUIState } from '../experiment-page/contexts/ExperimentPageUIStateContext';
import { ExperimentPageUIStateV2, RUNS_VISIBILITY_MODE } from '../experiment-page/models/ExperimentPageUIStateV2';
import { RunRowType } from '../experiment-page/utils/experimentPage.row-types';
import { RunsChartsSectionAccordion } from '../runs-charts/components/sections/RunsChartsSectionAccordion';
import { ReduxState } from 'redux-types';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { SearchIcon } from '@databricks/design-system';
import { Input } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { shouldEnableRunGrouping, shouldUseNewRunRowsVisibilityModel } from '../../../common/utils/FeatureUtils';
import { getRunGroupDisplayName, isRemainingRunsGroup } from '../experiment-page/utils/experimentPage.group-row-utils';
import { keyBy, values } from 'lodash';
import {
  type RunsChartsUIConfigurationSetter,
  RunsChartsUIConfigurationContextProvider,
  useUpdateRunsChartsUIConfiguration,
  useReorderRunsChartsFn,
  useInsertRunsChartsFn,
  useRemoveRunsChartFn,
  useConfirmChartCardConfigurationFn,
} from '../runs-charts/hooks/useRunsChartsUIConfiguration';
import { useToggleRowVisibilityCallback } from '../experiment-page/hooks/useToggleRowVisibilityCallback';
import { RunsChartsFullScreenModal } from '../runs-charts/components/RunsChartsFullScreenModal';

export interface RunsComparePropsV2 {
  comparedRuns: RunRowType[];
  isLoading: boolean;
  metricKeyList: string[];
  paramKeyList: string[];
  experimentTags: Record<string, KeyValueEntity>;
  compareRunCharts?: SerializedRunsChartsCardConfigCard[];
  compareRunSections?: ChartSectionConfig[];
  groupBy: string;
}

/**
 * Utility function: based on a run row coming from runs table, creates run data trace to be used in charts
 */
const createRunDataTrace = (
  run: RunRowType,
  latestMetricsByRunUuid: Record<string, MetricEntitiesByName>,
  paramsByRunUuid: Record<string, Record<string, KeyValueEntity>>,
) => ({
  uuid: run.runUuid,
  displayName: run.runInfo?.run_name || run.runUuid,
  runInfo: run.runInfo,
  metrics: latestMetricsByRunUuid[run.runUuid] || {},
  params: paramsByRunUuid[run.runUuid] || {},
  color: run.color,
  pinned: run.pinned,
  pinnable: run.pinnable,
  metricsHistory: {},
  belongsToGroup: run.runDateAndNestInfo?.belongsToGroup,
  hidden: run.hidden,
});

/**
 * Utility function: based on a group row coming from runs table, creates run group data trace to be used in charts
 */
const createGroupDataTrace = (
  run: RunRowType,
  allRunRows: RunRowType[],
  latestMetricsByRunUuid: Record<string, MetricEntitiesByName>,
  paramsByRunUuid: Record<string, Record<string, KeyValueEntity>>,
) => {
  // Latest aggregated metrics in groups does not contain step or timestamps.
  // For step, we're using maxStep which will help determine the chart type.
  // For timestamp, we use 0 as a placeholder.
  const metricsData = run.groupParentInfo?.aggregatedMetricData
    ? keyBy(
        values(run.groupParentInfo?.aggregatedMetricData).map(({ key, value, maxStep }) => ({
          key,
          value,
          step: maxStep,
          timestamp: 0,
        })),
        'key',
      )
    : {};
  return {
    uuid: run.rowUuid,
    displayName: getRunGroupDisplayName(run.groupParentInfo),
    groupParentInfo: run.groupParentInfo,
    metrics: metricsData,
    params: run.groupParentInfo?.aggregatedParamData || {},
    color: run.color,
    pinned: run.pinned,
    pinnable: run.pinnable,
    metricsHistory: {},
    hidden: run.hidden,
  };
};

/**
 * Component displaying comparison charts and differences (and in future artifacts) between experiment runs.
 * Intended to be mounted next to runs table.
 *
 * This component extracts params/metrics from redux store by itself for quicker access. However,
 * it needs a provided list of compared run entries using same model as runs table.
 */
export const RunsCompareV2Impl = ({
  isLoading,
  comparedRuns,
  metricKeyList,
  paramKeyList,
  experimentTags,
  compareRunCharts,
  compareRunSections,
  groupBy,
}: RunsComparePropsV2) => {
  // Updater function for the general experiment view UI state
  const updateUIState = useUpdateExperimentViewUIState();

  // Updater function for charts UI state
  const updateChartsUIState = useUpdateRunsChartsUIConfiguration();

  const { paramsByRunUuid, latestMetricsByRunUuid } = useSelector((state: ReduxState) => ({
    paramsByRunUuid: state.entities.paramsByRunUuid,
    latestMetricsByRunUuid: state.entities.latestMetricsByRunUuid,
  }));
  const { theme } = useDesignSystemTheme();
  const [initiallyLoaded, setInitiallyLoaded] = useState(false);
  const [configuredCardConfig, setConfiguredCardConfig] = useState<RunsChartsCardConfig | null>(null);
  const [search, setSearch] = useState('');
  const { formatMessage } = useIntl();

  const [fullScreenChart, setFullScreenChart] = useState<
    { config: RunsChartsCardConfig; title: string; subtitle: ReactNode } | undefined
  >(undefined);

  const addNewChartCard = (metricSectionId: string) => {
    return (type: RunsChartType) => {
      // TODO: pass existing runs data and get pre-configured initial setup for every chart type
      setConfiguredCardConfig(RunsChartsCardConfig.getEmptyChartCardByType(type, false, undefined, metricSectionId));
    };
  };

  const startEditChart = useCallback((chartCard: RunsChartsCardConfig) => {
    setConfiguredCardConfig(chartCard);
  }, []);

  useEffect(() => {
    if (!initiallyLoaded && !isLoading) {
      setInitiallyLoaded(true);
    }
  }, [initiallyLoaded, isLoading]);

  const primaryMetricKey = useMemo(() => {
    const automlEntry = experimentTags[AUTOML_EVALUATION_METRIC_TAG];
    const mlflowPrimaryEntry = experimentTags[MLFLOW_EXPERIMENT_PRIMARY_METRIC_NAME];
    return automlEntry?.value || mlflowPrimaryEntry?.value || metricKeyList[0] || '';
  }, [experimentTags, metricKeyList]);

  /**
   * Dataset generated for all charts in a single place
   * If we're using v2 chart improvements, we're using sampled metrics so we don't need to
   * enrich results with "useMultipleChartsMetricHistory" result
   */
  const chartData: RunsChartsRunData[] = useMemo(() => {
    if (!shouldEnableRunGrouping() || !groupBy) {
      return comparedRuns
        .filter((run) => run.runInfo)
        .filter((run) => shouldUseNewRunRowsVisibilityModel() || !run.hidden)
        .map<RunsChartsRunData>((run) => createRunDataTrace(run, latestMetricsByRunUuid, paramsByRunUuid));
    }

    const groupChartDataEntries = comparedRuns
      .filter((run) => shouldUseNewRunRowsVisibilityModel() || !run.hidden)
      .filter((run) => run.groupParentInfo && !isRemainingRunsGroup(run.groupParentInfo))
      .map<RunsChartsRunData>((group) =>
        createGroupDataTrace(group, comparedRuns, latestMetricsByRunUuid, paramsByRunUuid),
      );

    const remainingRuns = comparedRuns
      .filter((run) => shouldUseNewRunRowsVisibilityModel() || !run.hidden)
      .filter((run) => !run.groupParentInfo && !run.runDateAndNestInfo?.belongsToGroup)
      .map((run) => createRunDataTrace(run, latestMetricsByRunUuid, paramsByRunUuid));

    return [...groupChartDataEntries, ...remainingRuns];
  }, [groupBy, comparedRuns, latestMetricsByRunUuid, paramsByRunUuid]);

  const { isLoading: isMetricHistoryLoading } = useMultipleChartsMetricHistory(
    compareRunCharts || [],
    chartData,
    false,
  );

  // Set chart cards to the user-facing base config if there is no other information.
  useEffect(() => {
    if ((!compareRunSections || !compareRunCharts) && chartData.length > 0) {
      const { resultChartSet, resultSectionSet } = RunsChartsCardConfig.getBaseChartAndSectionConfigs({
        primaryMetricKey,
        runsData: chartData,
        useParallelCoordinatesChart: true,
      });
      updateChartsUIState((current) => ({
        ...current,
        compareRunCharts: resultChartSet,
        compareRunSections: resultSectionSet,
      }));
    }
  }, [compareRunCharts, compareRunSections, primaryMetricKey, chartData, updateChartsUIState]);

  /**
   * When chartData changes, we need to update the RunCharts with the latest charts
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
  }, [chartData, updateChartsUIState]);

  const onTogglePin = useCallback(
    (runUuid: string) => {
      updateUIState((existingFacets: ExperimentPageUIStateV2) => ({
        ...existingFacets,
        runsPinned: !existingFacets.runsPinned.includes(runUuid)
          ? [...existingFacets.runsPinned, runUuid]
          : existingFacets.runsPinned.filter((r) => r !== runUuid),
      }));
    },
    [updateUIState],
  );

  const toggleRunVisibility = useToggleRowVisibilityCallback(comparedRuns);

  const onHideRun = useCallback(
    (runUuid: string) => {
      if (shouldUseNewRunRowsVisibilityModel()) {
        toggleRunVisibility(RUNS_VISIBILITY_MODE.CUSTOM, runUuid);
        return;
      }
      updateUIState((existingFacets: ExperimentPageUIStateV2) => ({
        ...existingFacets,
        runsHidden: [...existingFacets.runsHidden, runUuid],
      }));
    },
    [updateUIState, toggleRunVisibility],
  );

  const confirmChartCardConfiguration = useConfirmChartCardConfigurationFn();

  const submitForm = (configuredCard: Partial<RunsChartsCardConfig>) => {
    confirmChartCardConfiguration(configuredCard);

    // Hide the modal
    setConfiguredCardConfig(null);
  };

  /**
   * Removes the chart from the compare run view.
   */
  const removeChart = useRemoveRunsChartFn();

  /**
   * Reorders the charts in the compare run view.
   */
  const reorderCharts = useReorderRunsChartsFn();

  /*
   * Inserts the source chart into the target group
   */
  const insertCharts = useInsertRunsChartsFn();

  /**
   * Data utilized by the tooltip system:
   * runs data and toggle pin callback
   */
  const tooltipContextValue = useMemo(
    () => ({ runs: chartData, onTogglePin, onHideRun }),
    [chartData, onHideRun, onTogglePin],
  );

  if (!initiallyLoaded) {
    return (
      <div
        css={{
          borderTop: `1px solid ${theme.colors.border}`,
          borderLeft: `1px solid ${theme.colors.border}`,

          // Let's cover 1 pixel of the grid's border for the sleek look
          marginLeft: -1,

          position: 'relative' as const,
          backgroundColor: theme.colors.backgroundSecondary,
          paddingLeft: theme.spacing.md,
          paddingRight: theme.spacing.md,
          paddingBottom: theme.spacing.md,
          zIndex: 1,
          overflowY: 'auto' as const,
        }}
      >
        <LegacySkeleton />
      </div>
    );
  }

  return (
    <div
      css={{
        borderTop: `1px solid ${theme.colors.border}`,
        borderLeft: `1px solid ${theme.colors.border}`,

        // Let's cover 1 pixel of the grid's border for the sleek look
        marginLeft: -1,

        position: 'relative' as const,
        backgroundColor: theme.colors.backgroundSecondary,
        paddingLeft: theme.spacing.md,
        paddingRight: theme.spacing.md,
        paddingBottom: theme.spacing.md,
        zIndex: 1,
        overflowY: 'auto' as const,
      }}
      data-testid="experiment-view-compare-runs-chart-area"
    >
      <div
        css={{
          paddingTop: theme.spacing.sm,
          paddingBottom: theme.spacing.sm,
        }}
      >
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
      </div>
      <RunsChartsTooltipWrapper contextData={tooltipContextValue} component={RunsChartsTooltipBody}>
        <DndProvider backend={HTML5Backend}>
          <RunsChartsSectionAccordion
            compareRunSections={compareRunSections}
            compareRunCharts={compareRunCharts}
            reorderCharts={reorderCharts}
            insertCharts={insertCharts}
            chartData={chartData}
            isMetricHistoryLoading={isMetricHistoryLoading}
            startEditChart={startEditChart}
            removeChart={removeChart}
            addNewChartCard={addNewChartCard}
            search={search}
            groupBy={groupBy}
            setFullScreenChart={setFullScreenChart}
          />
        </DndProvider>
      </RunsChartsTooltipWrapper>
      {configuredCardConfig && (
        <RunsChartsConfigureModal
          chartRunData={chartData}
          metricKeyList={metricKeyList}
          paramKeyList={paramKeyList}
          config={configuredCardConfig}
          onSubmit={submitForm}
          onCancel={() => setConfiguredCardConfig(null)}
          groupBy={groupBy}
        />
      )}
      <RunsChartsFullScreenModal
        fullScreenChart={fullScreenChart}
        onCancel={() => setFullScreenChart(undefined)}
        chartData={chartData}
        isMetricHistoryLoading={isMetricHistoryLoading}
        groupBy={groupBy}
        tooltipContextValue={tooltipContextValue}
        tooltipComponent={RunsChartsTooltipBody}
      />
    </div>
  );
};

export const RunsCompareV2 = (props: RunsComparePropsV2) => {
  // Updater function for the general experiment view UI state
  const updateUIState = useUpdateExperimentViewUIState();

  // An extracted partial updater function, responsible for setting charts UI state
  const updateChartsUIState = useCallback<(stateSetter: RunsChartsUIConfigurationSetter) => void>(
    (setter) => {
      updateUIState((state) => ({
        ...state,
        ...setter(state),
      }));
    },
    [updateUIState],
  );

  return (
    <RunsChartsUIConfigurationContextProvider updateChartsUIState={updateChartsUIState}>
      <RunsCompareV2Impl {...props} />
    </RunsChartsUIConfigurationContextProvider>
  );
};
