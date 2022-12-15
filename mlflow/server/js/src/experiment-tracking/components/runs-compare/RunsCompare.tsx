import {
  BarChartIcon,
  Button,
  ChartLineIcon,
  DropdownMenu,
  PlusIcon,
  Skeleton,
  SlidersIcon,
} from '@databricks/design-system';
import { Theme } from '@emotion/react';
import { useCallback, useEffect, useState } from 'react';
import { connect } from 'react-redux';
import type {
  CompareRunsChartSetup,
  ExperimentStoreEntities,
  KeyValueEntity,
  MetricEntitiesByName,
  UpdateExperimentSearchFacetsFn,
} from '../../types';
import { RunRowType } from '../experiment-page/utils/experimentPage.row-types';
import { RunsCompareCharts } from './RunsCompareCharts';
import { SearchExperimentRunsFacetsState } from '../experiment-page/models/SearchExperimentRunsFacetsState';
import { ConfigureChartModal } from './ConfigureChartModal';
import { getUUID } from '../../../common/utils/ActionUtils';

export interface RunsCompareProps {
  comparedRuns: RunRowType[];
  isLoading: boolean;
  metricKeyList: string[];
  paramKeyList: string[];
  searchFacetsState: SearchExperimentRunsFacetsState;
  updateSearchFacets: UpdateExperimentSearchFacetsFn;

  // Provided by redux connect().
  paramsByRunUuid: Record<string, Record<string, KeyValueEntity>>;
  metricsByRunUuid: Record<string, MetricEntitiesByName>;
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
}: RunsCompareProps) => {
  const [initiallyLoaded, setInitiallyLoaded] = useState(false);
  const [configureChartModalVisible, setConfigureChartModalVisible] = useState(false);

  useEffect(() => {
    if (!initiallyLoaded && !isLoading) {
      setInitiallyLoaded(true);
    }
  }, [initiallyLoaded, isLoading]);

  // TODO: implement various plot type creation
  const addNewPlot = useCallback(() => {
    setConfigureChartModalVisible(true);
  }, []);

  const submitForm = (formConfig: Pick<CompareRunsChartSetup, 'type' | 'metricKey'>) => {
    // TODO: implement validation
    const chartSetupToSave: CompareRunsChartSetup = {
      ...formConfig,
      uuid: getUUID(),
    };

    // Register new chart in the persistable state
    updateSearchFacets((current) => ({
      ...current,
      compareRunCharts: [...current.compareRunCharts, chartSetupToSave],
    }));

    // Hide the modal
    setConfigureChartModalVisible(false);
  };

  const removeChart = (configToDelete: CompareRunsChartSetup) => {
    updateSearchFacets((current) => ({
      ...current,
      compareRunCharts: current.compareRunCharts.filter(
        (setup) => setup.uuid !== configToDelete.uuid,
      ),
    }));
  };

  if (!initiallyLoaded) {
    return (
      <div css={styles.wrapper}>
        <Skeleton />
      </div>
    );
  }

  return (
    <div css={styles.wrapper}>
      <div css={styles.controlsWrapper}>
        <DropdownMenu.Root>
          <DropdownMenu.Trigger asChild>
            <Button icon={<PlusIcon />}>Add</Button>
          </DropdownMenu.Trigger>
          <DropdownMenu.Content align='end'>
            <DropdownMenu.Group>
              <DropdownMenu.Label>Advanced Comparison</DropdownMenu.Label>
              <DropdownMenu.Item onClick={addNewPlot}>
                <DropdownMenu.IconWrapper>
                  <ChartLineIcon />
                </DropdownMenu.IconWrapper>
                Scatter plot
              </DropdownMenu.Item>
              <DropdownMenu.Item onClick={addNewPlot}>
                <DropdownMenu.IconWrapper>
                  <SlidersIcon />
                </DropdownMenu.IconWrapper>
                Contour plot
              </DropdownMenu.Item>
            </DropdownMenu.Group>
            <DropdownMenu.Group>
              <DropdownMenu.Label>Metrics</DropdownMenu.Label>
              <DropdownMenu.Item onClick={addNewPlot}>
                <DropdownMenu.IconWrapper>
                  <BarChartIcon />
                </DropdownMenu.IconWrapper>
                (TODO) Metric 1
              </DropdownMenu.Item>
            </DropdownMenu.Group>
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      </div>
      <RunsCompareCharts
        comparedRuns={comparedRuns}
        chartsConfig={searchFacetsState.compareRunCharts}
        onRemoveChart={removeChart}
      />
      {configureChartModalVisible && (
        <ConfigureChartModal
          onSubmit={submitForm}
          onCancel={() => setConfigureChartModalVisible(false)}
        />
      )}
    </div>
  );
};

const styles = {
  controlsWrapper: (theme: Theme) => ({
    marginBottom: theme.spacing.md,
    display: 'flex' as const,
    justifyContent: 'flex-end',
  }),
  wrapper: (theme: Theme) => ({
    // Same height as "Show N matching runs" label.
    // Going to be fixed after switching to grid's fixed viewport height mode.

    // Let's cover 1 pixel of the grid's border for the sleek look
    marginLeft: -1,

    position: 'relative' as const,
    backgroundColor: theme.colors.backgroundSecondary,
    padding: theme.spacing.md,
    borderLeft: `1px solid ${theme.colors.border}`,
    zIndex: 1,
  }),
};

const mapStateToProps = ({ entities }: { entities: ExperimentStoreEntities }) => {
  const { paramsByRunUuid, latestMetricsByRunUuid } = entities;
  return { paramsByRunUuid, metricsByRunUuid: latestMetricsByRunUuid };
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
