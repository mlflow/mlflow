import { LegacySkeleton } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { connect } from 'react-redux';
import type {
  ExperimentStoreEntities,
  KeyValueEntity,
  MetricEntitiesByName,
  MetricHistoryByName,
  UpdateExperimentSearchFacetsFn,
} from '../../types';
import { RunRowType } from '../experiment-page/utils/experimentPage.row-types';
import { RunsCompareCardConfig } from './runs-compare.types';
import type { RunsCompareChartType } from './runs-compare.types';
import { RunsCompareAddChartMenu } from './RunsCompareAddChartMenu';
import { RunsCompareCharts } from './RunsCompareCharts';
import { SearchExperimentRunsFacetsState } from '../experiment-page/models/SearchExperimentRunsFacetsState';
import { RunsCompareConfigureModal } from './RunsCompareConfigureModal';
import { getUUID } from '../../../common/utils/ActionUtils';
import type { CompareChartRunData } from './charts/CompareRunsCharts.common';
import {
  AUTOML_EVALUATION_METRIC_TAG,
  MLFLOW_EXPERIMENT_PRIMARY_METRIC_NAME,
} from '../../constants';
import { RunsCompareTooltipBody } from './RunsCompareTooltipBody';
import { CompareRunsTooltipWrapper } from './hooks/useCompareRunsTooltip';
import { useMultipleChartsMetricHistory } from './hooks/useMultipleChartsMetricHistory';

export interface RunsCompareProps {
  comparedRuns: RunRowType[];
  isLoading: boolean;
  metricKeyList: string[];
  paramKeyList: string[];
  experimentTags: Record<string, KeyValueEntity>;
  searchFacetsState: SearchExperimentRunsFacetsState;
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
  searchFacetsState,
  updateSearchFacets,
  latestMetricsByRunUuid,
  metricsByRunUuid,
  paramsByRunUuid,
  metricKeyList,
  paramKeyList,
  experimentTags,
}: RunsCompareProps) => {
  const [initiallyLoaded, setInitiallyLoaded] = useState(false);
  const [configuredCardConfig, setConfiguredCardConfig] = useState<RunsCompareCardConfig | null>(
    null,
  );

  const addNewChartCard = useCallback((type: RunsCompareChartType) => {
    // TODO: pass existing runs data and get pre-configured initial setup for every chart type
    setConfiguredCardConfig(RunsCompareCardConfig.getEmptyChartCardByType(type));
  }, []);

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

  /**
   * Dataset generated for all charts in a single place
   */
  const chartRunData: CompareChartRunData[] = useMemo(
    () =>
      comparedRuns
        .filter((run) => !run.hidden)
        .map((run) => ({
          runInfo: run.runInfo,
          metrics: latestMetricsByRunUuid[run.runUuid] || {},
          params: paramsByRunUuid[run.runUuid] || {},
          color: run.color,
          pinned: run.pinned,
          pinnable: run.pinnable,
          metricsHistory: {},
        })),
    [comparedRuns, latestMetricsByRunUuid, paramsByRunUuid],
  );

  const { isLoading: isMetricHistoryLoading, chartRunDataWithHistory } =
    useMultipleChartsMetricHistory(
      searchFacetsState.compareRunCharts || [],
      chartRunData,
      metricsByRunUuid,
    );

  // Set chart cards to the user-facing base config if there is no other information.
  useEffect(() => {
    if (!searchFacetsState.compareRunCharts && chartRunDataWithHistory.length > 0) {
      updateSearchFacets(
        (current) => ({
          ...current,
          compareRunCharts: RunsCompareCardConfig.getBaseChartConfigs(
            primaryMetricKey,
            chartRunDataWithHistory[0].metrics,
          ),
        }),
        { replaceHistory: true },
      );
    }
  }, [
    searchFacetsState.compareRunCharts,
    primaryMetricKey,
    updateSearchFacets,
    chartRunDataWithHistory,
  ]);

  const onTogglePin = useCallback(
    (runUuid: string) => {
      updateSearchFacets((existingFacets) => ({
        ...existingFacets,
        runsPinned: !existingFacets.runsPinned.includes(runUuid)
          ? [...existingFacets.runsPinned, runUuid]
          : existingFacets.runsPinned.filter((r) => r !== runUuid),
      }));
    },
    [updateSearchFacets],
  );

  const onHideRun = useCallback(
    (runUuid: string) => {
      updateSearchFacets((existingFacets) => ({
        ...existingFacets,
        runsHidden: [...existingFacets.runsHidden, runUuid],
      }));
    },
    [updateSearchFacets],
  );

  const submitForm = (configuredCard: Partial<RunsCompareCardConfig>) => {
    // TODO: implement validation
    const serializedCard = RunsCompareCardConfig.serialize({
      ...configuredCard,
      uuid: getUUID(),
    });

    // Creating new chart
    if (!configuredCard.uuid) {
      updateSearchFacets((current) => ({
        ...current,
        // This condition ensures that chart collection will remain undefined if not set previously
        compareRunCharts: current.compareRunCharts && [...current.compareRunCharts, serializedCard],
      }));
    } /* Editing existing chart */ else {
      updateSearchFacets((current) => ({
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
    updateSearchFacets((current) => ({
      ...current,
      compareRunCharts: current.compareRunCharts?.filter(
        (setup) => setup.uuid !== configToDelete.uuid,
      ),
    }));
  };

  /**
   * Data utilized by the tooltip system:
   * runs data and toggle pin callback
   */
  const tooltipContextValue = useMemo(
    () => ({ runs: chartRunDataWithHistory, onTogglePin, onHideRun }),
    [chartRunDataWithHistory, onHideRun, onTogglePin],
  );

  if (!initiallyLoaded) {
    return (
      <div css={styles.wrapper}>
        <LegacySkeleton />
      </div>
    );
  }

  return (
    <div css={styles.wrapper} data-testid='experiment-view-compare-runs-chart-area'>
      <div css={styles.controlsWrapper}>
        <RunsCompareAddChartMenu onAddChart={addNewChartCard} />
      </div>
      <CompareRunsTooltipWrapper
        contextData={tooltipContextValue}
        component={RunsCompareTooltipBody}
      >
        <RunsCompareCharts
          chartRunData={chartRunDataWithHistory}
          cardsConfig={searchFacetsState.compareRunCharts || []}
          onStartEditChart={startEditChart}
          onRemoveChart={removeChart}
          isMetricHistoryLoading={isMetricHistoryLoading}
        />
      </CompareRunsTooltipWrapper>
      {configuredCardConfig && (
        <RunsCompareConfigureModal
          chartRunData={chartRunDataWithHistory}
          metricKeyList={metricKeyList}
          paramKeyList={paramKeyList}
          config={configuredCardConfig}
          onSubmit={submitForm}
          onCancel={() => setConfiguredCardConfig(null)}
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
