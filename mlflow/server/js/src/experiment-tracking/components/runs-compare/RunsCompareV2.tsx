import { LegacySkeleton, useDesignSystemTheme } from '@databricks/design-system';
import type { Theme } from '@emotion/react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
import { RunsCompareCardConfig } from './runs-compare.types';
import type { RunsCompareChartType, SerializedRunsCompareCardConfigCard } from './runs-compare.types';
import { RunsCompareConfigureModal } from './RunsCompareConfigureModal';
import { getUUID } from '../../../common/utils/ActionUtils';
import type { RunsChartsRunData } from '../runs-charts/components/RunsCharts.common';
import { AUTOML_EVALUATION_METRIC_TAG, MLFLOW_EXPERIMENT_PRIMARY_METRIC_NAME } from '../../constants';
import { RunsCompareTooltipBody } from './RunsCompareTooltipBody';
import { RunsChartsTooltipWrapper } from '../runs-charts/hooks/useRunsChartsTooltip';
import { useMultipleChartsMetricHistory } from './hooks/useMultipleChartsMetricHistory';
import { useUpdateExperimentViewUIState } from '../experiment-page/contexts/ExperimentPageUIStateContext';
import { ExperimentPageUIStateV2 } from '../experiment-page/models/ExperimentPageUIStateV2';
import { RunRowType } from '../experiment-page/utils/experimentPage.row-types';
import { RunsCompareSectionAccordion } from './sections/RunsCompareSectionAccordion';
import { ReduxState } from 'redux-types';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { SearchIcon } from '@databricks/design-system';
import { Input } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import {
  shouldEnableDeepLearningUIPhase2,
  shouldEnableShareExperimentViewByTags,
} from '../../../common/utils/FeatureUtils';
import { getRunGroupDisplayName, isRemainingRunsGroup } from '../experiment-page/utils/experimentPage.group-row-utils';
import { keyBy, values } from 'lodash';

export interface RunsComparePropsV2 {
  comparedRuns: RunRowType[];
  isLoading: boolean;
  metricKeyList: string[];
  paramKeyList: string[];
  experimentTags: Record<string, KeyValueEntity>;
  compareRunCharts?: SerializedRunsCompareCardConfigCard[];
  compareRunSections?: ChartSectionConfig[];
  groupBy: string;
}

/**
 * Component displaying comparison charts and differences (and in future artifacts) between experiment runs.
 * Intended to be mounted next to runs table.
 *
 * This component extracts params/metrics from redux store by itself for quicker access. However,
 * it needs a provided list of compared run entries using same model as runs table.
 */
export const RunsCompareV2 = ({
  isLoading,
  comparedRuns,
  metricKeyList,
  paramKeyList,
  experimentTags,
  compareRunCharts,
  compareRunSections,
  groupBy,
}: RunsComparePropsV2) => {
  const updateUIState = useUpdateExperimentViewUIState();
  const runGroupingEnabled = shouldEnableShareExperimentViewByTags() && shouldEnableDeepLearningUIPhase2();

  const { paramsByRunUuid, latestMetricsByRunUuid } = useSelector((state: ReduxState) => ({
    paramsByRunUuid: state.entities.paramsByRunUuid,
    latestMetricsByRunUuid: state.entities.latestMetricsByRunUuid,
  }));
  const { theme } = useDesignSystemTheme();
  const [initiallyLoaded, setInitiallyLoaded] = useState(false);
  const [configuredCardConfig, setConfiguredCardConfig] = useState<RunsCompareCardConfig | null>(null);
  const [search, setSearch] = useState('');
  const { formatMessage } = useIntl();

  const addNewChartCard = (metricSectionId: string) => {
    return (type: RunsCompareChartType) => {
      // TODO: pass existing runs data and get pre-configured initial setup for every chart type
      setConfiguredCardConfig(RunsCompareCardConfig.getEmptyChartCardByType(type, false, undefined, metricSectionId));
    };
  };

  const startEditChart = useCallback((chartCard: RunsCompareCardConfig) => {
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

  const containsGroupedRuns = comparedRuns.some((run) => run.groupParentInfo);

  /**
   * Dataset generated for all charts in a single place
   * If we're using v2 chart improvements, we're using sampled metrics so we don't need to
   * enrich results with "useMultipleChartsMetricHistory" result
   */
  const chartData: RunsChartsRunData[] = useMemo(() => {
    if (!runGroupingEnabled || !groupBy) {
      return comparedRuns
        .filter((run) => !run.hidden && run.runInfo)
        .map<RunsChartsRunData>((run) => ({
          uuid: run.runUuid,
          displayName: run.runInfo?.run_name || run.runUuid,
          runInfo: run.runInfo,
          metrics: latestMetricsByRunUuid[run.runUuid] || {},
          params: paramsByRunUuid[run.runUuid] || {},
          color: run.color,
          pinned: run.pinned,
          pinnable: run.pinnable,
          metricsHistory: {},
        }));
    }

    const createGroupDataTrace = (run: RunRowType) => {
      // Latest aggregated metrics in groups does not contain step or timestamps, we need to add empty values here
      const metricsData = run.groupParentInfo?.aggregatedMetricData
        ? keyBy(
            values(run.groupParentInfo?.aggregatedMetricData).map(({ key, value }) => ({
              key,
              value,
              step: 0,
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
      };
    };

    const groupChartDataEntries = comparedRuns
      .filter((run) => !run.hidden)
      .filter((run) => run.groupParentInfo && !isRemainingRunsGroup(run.groupParentInfo))
      .map<RunsChartsRunData>(createGroupDataTrace);

    const remainingRuns = comparedRuns
      .filter((run) => !run.groupParentInfo && !run.hidden && !run.runDateAndNestInfo?.belongsToGroup)
      .map((run) => ({
        uuid: run.runUuid,
        displayName: run.runInfo?.run_name || run.runUuid,
        runInfo: run.runInfo,
        metrics: latestMetricsByRunUuid[run.runUuid] || {},
        params: paramsByRunUuid[run.runUuid] || {},
        belongsToGroup: run.runDateAndNestInfo?.belongsToGroup,
        color: run.color,
        pinned: run.pinned,
        pinnable: run.pinnable,
        metricsHistory: {},
      }));

    return [...groupChartDataEntries, ...remainingRuns];
  }, [groupBy, comparedRuns, latestMetricsByRunUuid, paramsByRunUuid, runGroupingEnabled]);

  const { isLoading: isMetricHistoryLoading } = useMultipleChartsMetricHistory(
    compareRunCharts || [],
    chartData,
    false,
  );

  // Set chart cards to the user-facing base config if there is no other information.
  useEffect(() => {
    if ((!compareRunSections || !compareRunCharts) && chartData.length > 0) {
      const { resultChartSet, resultSectionSet } = RunsCompareCardConfig.getBaseChartAndSectionConfigs(
        primaryMetricKey,
        chartData,
      );
      updateUIState((current) => ({
        ...current,
        compareRunCharts: resultChartSet,
        compareRunSections: resultSectionSet,
      }));
    }
  }, [compareRunCharts, compareRunSections, primaryMetricKey, chartData, updateUIState]);

  /**
   * When chartData changes, we need to update the RunCharts with the latest charts
   */
  useEffect(() => {
    updateUIState((current) => {
      if (!current.compareRunCharts || !current.compareRunSections) {
        return current;
      }

      const { resultChartSet, resultSectionSet, isResultUpdated } = RunsCompareCardConfig.updateChartAndSectionConfigs(
        current.compareRunCharts,
        current.compareRunSections,
        chartData,
        current.isAccordionReordered,
      );

      if (!isResultUpdated) {
        return current;
      }
      return {
        ...current,
        compareRunCharts: resultChartSet,
        compareRunSections: resultSectionSet,
      };
    });
  }, [chartData, updateUIState]);

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

  const onHideRun = useCallback(
    (runUuid: string) => {
      updateUIState((existingFacets: ExperimentPageUIStateV2) => ({
        ...existingFacets,
        runsHidden: [...existingFacets.runsHidden, runUuid],
      }));
    },
    [updateUIState],
  );

  const submitForm = (configuredCard: Partial<RunsCompareCardConfig>) => {
    // TODO: implement validation
    const serializedCard = RunsCompareCardConfig.serialize({
      ...configuredCard,
      uuid: getUUID(),
    });

    // Creating new chart
    if (!configuredCard.uuid) {
      updateUIState((current: ExperimentPageUIStateV2) => ({
        ...current,
        // This condition ensures that chart collection will remain undefined if not set previously
        compareRunCharts: current.compareRunCharts && [...current.compareRunCharts, serializedCard],
      }));
    } /* Editing existing chart */ else {
      updateUIState((current: ExperimentPageUIStateV2) => ({
        ...current,
        compareRunCharts: current.compareRunCharts?.map((existingChartCard) => {
          if (existingChartCard.uuid === configuredCard.uuid) {
            return serializedCard;
          }
          return existingChartCard;
        }),
      }));
    }

    // Hide the modal
    setConfiguredCardConfig(null);
  };

  const removeChart = (configToDelete: RunsCompareCardConfig) => {
    updateUIState((current: ExperimentPageUIStateV2) => ({
      ...current,
      compareRunCharts: configToDelete.isGenerated
        ? current.compareRunCharts?.map((setup) =>
            setup.uuid === configToDelete.uuid ? { ...setup, deleted: true } : setup,
          )
        : current.compareRunCharts?.filter((setup) => setup.uuid !== configToDelete.uuid),
    }));
  };

  /**
   * Reorders the charts in the compare run view.
   */
  const reorderCharts = (sourceChartUuid: string, targetChartUuid: string) => {
    updateUIState((current) => {
      const newChartsOrder = current.compareRunCharts?.slice();
      const newSectionsState = current.compareRunSections?.slice();
      if (!newChartsOrder || !newSectionsState) {
        return current;
      }

      const indexSource = newChartsOrder.findIndex((c) => c.uuid === sourceChartUuid);
      const indexTarget = newChartsOrder.findIndex((c) => c.uuid === targetChartUuid);

      // If one of the charts is not found, do nothing
      if (indexSource < 0 || indexTarget < 0) {
        return current;
      }

      const sourceChart = newChartsOrder[indexSource];
      const targetChart = newChartsOrder[indexTarget];

      const isSameMetricSection = targetChart.metricSectionId === sourceChart.metricSectionId;

      // Update the sections to indicate that the charts have been reordered
      const sourceSectionIdx = newSectionsState.findIndex((c) => c.uuid === sourceChart.metricSectionId);
      const targetSectionIdx = newSectionsState.findIndex((c) => c.uuid === targetChart.metricSectionId);
      newSectionsState.splice(sourceSectionIdx, 1, { ...newSectionsState[sourceSectionIdx], isReordered: true });
      newSectionsState.splice(targetSectionIdx, 1, { ...newSectionsState[targetSectionIdx], isReordered: true });

      // Set new chart metric group
      const newSourceChart = { ...sourceChart };
      newSourceChart.metricSectionId = targetChart.metricSectionId;

      // Remove the source graph from array
      newChartsOrder.splice(indexSource, 1);
      if (!isSameMetricSection) {
        // Insert the source graph into target
        newChartsOrder.splice(
          newChartsOrder.findIndex((c) => c.uuid === targetChartUuid),
          0,
          newSourceChart,
        );
      } else {
        // The indexTarget is not neccessarily the target now, but it will work as the insert index
        newChartsOrder.splice(indexTarget, 0, newSourceChart);
      }

      return {
        ...current,
        compareRunCharts: newChartsOrder,
        compareRunSections: newSectionsState,
      };
    });
  };

  /*
   * Inserts the source chart into the target group
   */
  const insertCharts = (sourceChartUuid: string, targetSectionId: string) => {
    updateUIState((current) => {
      const newChartsOrder = current.compareRunCharts?.slice();
      const newSectionsState = current.compareRunSections?.slice();
      if (!newChartsOrder || !newSectionsState) {
        return current;
      }

      const indexSource = newChartsOrder.findIndex((c) => c.uuid === sourceChartUuid);
      if (indexSource < 0) {
        return current;
      }
      const sourceChart = newChartsOrder[indexSource];
      // Set new chart metric group
      const newSourceChart = { ...sourceChart };
      newSourceChart.metricSectionId = targetSectionId;

      // Update the sections to indicate that the charts have been reordered
      const sourceSectionIdx = newSectionsState.findIndex((c) => c.uuid === sourceChart.metricSectionId);
      const targetSectionIdx = newSectionsState.findIndex((c) => c.uuid === targetSectionId);
      newSectionsState.splice(sourceSectionIdx, 1, { ...newSectionsState[sourceSectionIdx], isReordered: true });
      newSectionsState.splice(targetSectionIdx, 1, { ...newSectionsState[targetSectionIdx], isReordered: true });

      // Remove the source graph from array and append
      newChartsOrder.splice(indexSource, 1);
      newChartsOrder.push(newSourceChart);

      return {
        ...current,
        compareRunCharts: newChartsOrder,
        compareRunSections: newSectionsState,
      };
    });
  };

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

      <RunsChartsTooltipWrapper contextData={tooltipContextValue} component={RunsCompareTooltipBody}>
        <DndProvider backend={HTML5Backend}>
          <RunsCompareSectionAccordion
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
          />
        </DndProvider>
      </RunsChartsTooltipWrapper>
      {configuredCardConfig && (
        <RunsCompareConfigureModal
          chartRunData={chartData}
          metricKeyList={metricKeyList}
          paramKeyList={paramKeyList}
          config={configuredCardConfig}
          onSubmit={submitForm}
          onCancel={() => setConfiguredCardConfig(null)}
          groupBy={groupBy}
        />
      )}
    </div>
  );
};
