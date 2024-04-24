import { LegacySkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { connect } from 'react-redux';
import type {
  ExperimentStoreEntities,
  KeyValueEntity,
  MetricEntitiesByName,
  MetricHistoryByName,
  RunInfoEntity,
  UpdateExperimentSearchFacetsFn,
} from '../../types';
import { RunRowType } from '../experiment-page/utils/experimentPage.row-types';
import { RunsChartsCardConfig } from '../runs-charts/runs-charts.types';
import type { RunsChartType, SerializedRunsChartsCardConfigCard } from '../runs-charts/runs-charts.types';
import { RunsChartsAddChartMenu } from '../runs-charts/components/RunsChartsAddChartMenu';
import { RunsCompareCharts } from './RunsCompareCharts';
import { RunsChartsConfigureModal } from '../runs-charts/components/RunsChartsConfigureModal';
import { getUUID } from '../../../common/utils/ActionUtils';
import type { RunsChartsRunData } from '../runs-charts/components/RunsCharts.common';
import { AUTOML_EVALUATION_METRIC_TAG, MLFLOW_EXPERIMENT_PRIMARY_METRIC_NAME } from '../../constants';
import { RunsChartsTooltipBody } from '../runs-charts/components/RunsChartsTooltipBody';
import { RunsChartsTooltipWrapper } from '../runs-charts/hooks/useRunsChartsTooltip';
import { useMultipleChartsMetricHistory } from './hooks/useMultipleChartsMetricHistory';
import { shouldEnableDeepLearningUI, shouldEnableShareExperimentViewByTags } from '../../../common/utils/FeatureUtils';
import { useUpdateExperimentViewUIState } from '../experiment-page/contexts/ExperimentPageUIStateContext';
import { ExperimentPageUIStateV2 } from '../experiment-page/models/ExperimentPageUIStateV2';

export interface RunsCompareProps {
  comparedRuns: RunRowType[];
  isLoading: boolean;
  metricKeyList: string[];
  paramKeyList: string[];
  experimentTags: Record<string, KeyValueEntity>;
  compareRunCharts?: SerializedRunsChartsCardConfigCard[];
  updateSearchFacets: UpdateExperimentSearchFacetsFn;
  // Provided by redux connect().
  paramsByRunUuid: Record<string, Record<string, KeyValueEntity>>;
  latestMetricsByRunUuid: Record<string, MetricEntitiesByName>;
  metricsByRunUuid: Record<string, MetricHistoryByName>;
}

/**
 * Component displaying comparison charts and differences (and in future artifacts) between experiment runs.
 * Intended to be mounted next to runs table.
 *
 * This component extracts params/metrics from redux store by itself for quicker access, however
 * it needs a provided list of compared run entries using same model as runs table.
 */
export const RunsCompareImpl = ({
  comparedRuns,
  isLoading,
  compareRunCharts,
  updateSearchFacets,
  latestMetricsByRunUuid,
  metricsByRunUuid,
  paramsByRunUuid,
  metricKeyList,
  paramKeyList,
  experimentTags,
}: RunsCompareProps) => {
  const usingV2ChartImprovements = shouldEnableDeepLearningUI();
  const usingNewViewStateModel = shouldEnableShareExperimentViewByTags();
  const updateUIState = useUpdateExperimentViewUIState();
  const stateSetterFn = usingNewViewStateModel ? updateUIState : updateSearchFacets;

  const { theme } = useDesignSystemTheme();
  const [initiallyLoaded, setInitiallyLoaded] = useState(false);
  const [configuredCardConfig, setConfiguredCardConfig] = useState<RunsChartsCardConfig | null>(null);

  const addNewChartCard = useCallback((type: RunsChartType) => {
    // TODO: pass existing runs data and get pre-configured initial setup for every chart type
    setConfiguredCardConfig(RunsChartsCardConfig.getEmptyChartCardByType(type, true));
  }, []);

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
   */
  const chartRunData: RunsChartsRunData[] = useMemo(
    () =>
      comparedRuns
        .filter((run) => !run.hidden)
        .map((run) => ({
          // At this point, we're certain that runInfo is present in the run row
          displayName: (run.runInfo as RunInfoEntity).run_name || '',
          runInfo: run.runInfo,
          metrics: (run.runUuid && latestMetricsByRunUuid[run.runUuid]) || {},
          params: (run.runUuid && paramsByRunUuid[run.runUuid]) || {},
          color: run.color,
          pinned: run.pinned,
          pinnable: run.pinnable,
          metricsHistory: {},
          uuid: run.rowUuid,
        })),
    [comparedRuns, latestMetricsByRunUuid, paramsByRunUuid],
  );

  const { isLoading: isMetricHistoryLoading, chartRunDataWithHistory: chartRunDataWithUnsampledHistory } =
    useMultipleChartsMetricHistory(compareRunCharts || [], chartRunData, !usingV2ChartImprovements);

  // If we're using v2 chart improvements, we're using sampled metrics so we don't need to
  // enrich results with "useMultipleChartsMetricHistory" result
  const chartData = usingV2ChartImprovements ? chartRunData : chartRunDataWithUnsampledHistory;

  // Set chart cards to the user-facing base config if there is no other information.
  useEffect(() => {
    if (!compareRunCharts && chartData.length > 0) {
      if (usingNewViewStateModel) {
        updateUIState((current) => ({
          ...current,
          compareRunCharts: RunsChartsCardConfig.getBaseChartConfigs(primaryMetricKey, chartData),
        }));
      } else {
        updateSearchFacets(
          (current) => ({
            ...current,
            compareRunCharts: RunsChartsCardConfig.getBaseChartConfigs(primaryMetricKey, chartData),
          }),
          { replaceHistory: true },
        );
      }
    }
  }, [compareRunCharts, primaryMetricKey, updateSearchFacets, chartData, usingNewViewStateModel, updateUIState]);

  const onTogglePin = useCallback(
    (runUuid: string) => {
      stateSetterFn((existingFacets: ExperimentPageUIStateV2) => ({
        ...existingFacets,
        runsPinned: !existingFacets.runsPinned.includes(runUuid)
          ? [...existingFacets.runsPinned, runUuid]
          : existingFacets.runsPinned.filter((r) => r !== runUuid),
      }));
    },
    [stateSetterFn],
  );

  const onHideRun = useCallback(
    (runUuid: string) => {
      stateSetterFn((existingFacets: ExperimentPageUIStateV2) => ({
        ...existingFacets,
        runsHidden: [...existingFacets.runsHidden, runUuid],
      }));
    },
    [stateSetterFn],
  );

  const submitForm = (configuredCard: Partial<RunsChartsCardConfig>) => {
    // TODO: implement validation
    const serializedCard = RunsChartsCardConfig.serialize({
      ...configuredCard,
      uuid: getUUID(),
    });

    // Creating new chart
    if (!configuredCard.uuid) {
      stateSetterFn((current: ExperimentPageUIStateV2) => ({
        ...current,
        // This condition ensures that chart collection will remain undefined if not set previously
        compareRunCharts: current.compareRunCharts && [...current.compareRunCharts, serializedCard],
      }));
    } /* Editing existing chart */ else {
      stateSetterFn((current: ExperimentPageUIStateV2) => ({
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

  const removeChart = (configToDelete: RunsChartsCardConfig) => {
    stateSetterFn((current: ExperimentPageUIStateV2) => ({
      ...current,
      compareRunCharts: current.compareRunCharts?.filter((setup) => setup.uuid !== configToDelete.uuid),
    }));
  };

  /**
   * Reorders the charts in the compare run view by swapping the positions of two charts.
   */
  const reorderCharts = (sourceChartUuid: string, targetChartUuid: string) => {
    stateSetterFn((current: ExperimentPageUIStateV2) => {
      const newChartsOrder = current.compareRunCharts?.slice();
      if (!newChartsOrder) {
        return current;
      }

      const indexSource = newChartsOrder.findIndex((c) => c.uuid === sourceChartUuid);
      const indexTarget = newChartsOrder.findIndex((c) => c.uuid === targetChartUuid);

      // If one of the charts is not found, do nothing
      if (indexSource < 0 || indexTarget < 0) {
        return current;
      }

      // Swap the charts
      [newChartsOrder[indexSource], newChartsOrder[indexTarget]] = [
        newChartsOrder[indexTarget],
        newChartsOrder[indexSource],
      ];

      return {
        ...current,
        compareRunCharts: newChartsOrder,
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
      <div css={styles.wrapper(theme)}>
        <LegacySkeleton />
      </div>
    );
  }

  return (
    <div css={styles.wrapper(theme)} data-testid="experiment-view-compare-runs-chart-area">
      <div css={styles.controlsWrapper(theme)}>
        <RunsChartsAddChartMenu onAddChart={addNewChartCard} />
      </div>
      <RunsChartsTooltipWrapper contextData={tooltipContextValue} component={RunsChartsTooltipBody}>
        <RunsCompareCharts
          chartRunData={chartData}
          cardsConfig={compareRunCharts || []}
          onStartEditChart={startEditChart}
          onRemoveChart={removeChart}
          isMetricHistoryLoading={isMetricHistoryLoading}
          onReorderCharts={reorderCharts}
          groupBy=""
        />
      </RunsChartsTooltipWrapper>
      {configuredCardConfig && (
        <RunsChartsConfigureModal
          chartRunData={chartData}
          metricKeyList={metricKeyList}
          paramKeyList={paramKeyList}
          config={configuredCardConfig}
          onSubmit={submitForm}
          onCancel={() => setConfiguredCardConfig(null)}
          groupBy=""
        />
      )}
    </div>
  );
};

const styles = {
  controlsWrapper: (theme: Theme) => ({
    position: 'sticky' as const,
    top: 0,
    marginBottom: theme.spacing.md,
    display: 'flex' as const,
    justifyContent: 'flex-end',
    zIndex: 2,
    backgroundColor: theme.colors.backgroundSecondary,
    paddingTop: theme.spacing.md,
    paddingBottom: theme.spacing.md,
  }),
  wrapper: (theme: Theme) => ({
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
  }),
};

const mapStateToProps = ({ entities }: { entities: ExperimentStoreEntities }) => {
  const { paramsByRunUuid, latestMetricsByRunUuid, metricsByRunUuid } = entities;
  return { paramsByRunUuid, latestMetricsByRunUuid, metricsByRunUuid };
};

export const RunsCompare = connect(
  mapStateToProps,
  // mapDispatchToProps function (not provided):
  undefined,
  // mergeProps function (not provided):
  undefined,
  {
    // We're interested only in "entities" sub-tree so we won't
    // re-render on other state changes (e.g. API request IDs)
    areStatesEqual: (nextState, prevState) => nextState.entities === prevState.entities,
  },
)(RunsCompareImpl);
