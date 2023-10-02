import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { connect } from 'react-redux';
import { LegacySkeleton, useLegacyNotification } from '@databricks/design-system';
import {
  DatasetSummary,
  ExperimentEntity,
  ExperimentStoreEntities,
  LIFECYCLE_FILTER,
  MODEL_VERSION_FILTER,
  RunDatasetWithTags,
  UpdateExperimentViewStateFn,
} from '../../../../types';
import {
  experimentRunsSelector,
  ExperimentRunsSelectorParams,
  ExperimentRunsSelectorResult,
} from '../../utils/experimentRuns.selector';
import { ExperimentViewRunsControls } from './ExperimentViewRunsControls';
import { ExperimentViewRunsTable } from './ExperimentViewRunsTable';

import { searchModelVersionsApi } from '../../../../../model-registry/actions';
import { loadMoreRunsApi, searchRunsApi, searchRunsPayload } from '../../../../actions';
import { GetExperimentRunsContextProvider } from '../../contexts/GetExperimentRunsContext';
import { useFetchExperimentRuns } from '../../hooks/useFetchExperimentRuns';
import { SearchExperimentRunsViewState } from '../../models/SearchExperimentRunsViewState';
import Utils from '../../../../../common/utils/Utils';
import {
  ATTRIBUTE_COLUMN_SORT_KEY,
  MLFLOW_RUN_TYPE_TAG,
  MLFLOW_RUN_TYPE_VALUE_EVALUATION,
} from '../../../../constants';
import { RunRowType } from '../../utils/experimentPage.row-types';
import { prepareRunsGridData } from '../../utils/experimentPage.row-utils';
import { RunsCompare } from '../../../runs-compare/RunsCompare';
import { useFetchedRunsNotification } from '../../hooks/useFetchedRunsNotification';
import { DatasetWithRunType, ExperimentViewDatasetDrawer } from './ExperimentViewDatasetDrawer';
import { useExperimentViewLocalStore } from '../../hooks/useExperimentViewLocalStore';
import { useAutoExpandRunRows } from '../../hooks/useAutoExpandRunRows';
import { EvaluationArtifactCompareView } from '../../../evaluation-artifacts-compare/EvaluationArtifactCompareView';
import { shouldEnableArtifactBasedEvaluation } from '../../../../../common/utils/FeatureUtils';
import { CreateNewRunContextProvider } from '../../hooks/useCreateNewRun';
import { useChartViewByDefault } from '../../hooks/useChartViewByDefault';

export interface ExperimentViewRunsOwnProps {
  isLoading: boolean;
  experiments: ExperimentEntity[];
  modelVersionFilter?: MODEL_VERSION_FILTER;
  lifecycleFilter?: LIFECYCLE_FILTER;
  datasetsFilter?: DatasetSummary[];
  onMaximizedChange?: (newIsMaximized: boolean) => void;
}

export interface ExperimentViewRunsProps extends ExperimentViewRunsOwnProps {
  runsData: ExperimentRunsSelectorResult;
}

/**
 * Creates time with milliseconds set to zero, usable in calculating
 * relative time
 */
const createCurrentTime = () => {
  const mountTime = new Date();
  mountTime.setMilliseconds(0);
  return mountTime;
};

export const ExperimentViewRunsImpl = React.memo((props: ExperimentViewRunsProps) => {
  const { experiments, runsData, onMaximizedChange } = props;

  // Persistable sort/filter model state is taken from the context
  const {
    searchFacetsState,
    updateSearchFacets,
    isLoadingRuns,
    loadMoreRuns,
    moreRunsAvailable,
    isPristine,
    requestError,
  } = useFetchExperimentRuns();

  const [visibleRuns, setVisibleRuns] = useState<RunRowType[]>([]);

  // Non-persistable view model state is being created locally
  const [viewState, setViewState] = useState(new SearchExperimentRunsViewState());

  const { experiment_id } = experiments[0];
  const expandRowsStore = useExperimentViewLocalStore(experiment_id);
  const [expandRows, updateExpandRows] = useState<boolean>(
    expandRowsStore.getItem('expandRows') === 'true' ?? false,
  );

  useEffect(() => {
    expandRowsStore.setItem('expandRows', expandRows);
  }, [expandRows, expandRowsStore]);

  const {
    paramKeyList,
    metricKeyList,
    tagsList,
    modelVersionsByRunUuid,
    paramsList,
    metricsList,
    runInfos,
    runUuidsMatchingFilter,
    datasetsList,
  } = runsData;

  /**
   * Create a list of run infos with assigned metrics, params and tags
   */
  const runData = useMemo(
    () =>
      runInfos.map((runInfo, index) => ({
        runInfo,
        params: paramsList[index],
        metrics: metricsList[index],
        tags: tagsList[index],
        datasets: datasetsList[index],
      })),
    [datasetsList, metricsList, paramsList, runInfos, tagsList],
  );

  const {
    orderByKey,
    searchFilter,
    runsExpanded,
    runsPinned,
    runsHidden,
    compareRunsMode,
    datasetsFilter,
  } = searchFacetsState;

  const isComparingRuns = compareRunsMode !== undefined;

  // Default to chart view if metrics are available
  useChartViewByDefault(isLoadingRuns, metricKeyList, updateSearchFacets);
  // Automatically expand parent runs if necessary
  useAutoExpandRunRows(runData, visibleRuns, isPristine, updateSearchFacets, runsExpanded);

  const updateViewState = useCallback<UpdateExperimentViewStateFn>(
    (newPartialViewState) =>
      setViewState((currentViewState) => ({ ...currentViewState, ...newPartialViewState })),
    [],
  );

  const addColumnClicked = useCallback(() => {
    updateViewState({ columnSelectorVisible: true });
  }, [updateViewState]);

  const shouldNestChildrenAndFetchParents = useMemo(
    () => (!orderByKey && !searchFilter) || orderByKey === ATTRIBUTE_COLUMN_SORT_KEY.DATE,
    [orderByKey, searchFilter],
  );

  // Value used a reference for the "date" column
  const [referenceTime, setReferenceTime] = useState(createCurrentTime);

  // We're setting new reference date only when new runs data package has arrived
  useEffect(() => {
    setReferenceTime(createCurrentTime);
  }, [runInfos]);

  const filteredTagKeys = useMemo(() => Utils.getVisibleTagKeyList(tagsList), [tagsList]);

  const [isDrawerOpen, setIsDrawerOpen] = useState<boolean>(false);
  const [selectedDatasetWithRun, setSelectedDatasetWithRun] = useState<DatasetWithRunType>();

  useEffect(() => {
    if (isLoadingRuns) {
      return;
    }
    const runs = prepareRunsGridData({
      experiments,
      paramKeyList,
      metricKeyList,
      modelVersionsByRunUuid,
      runsExpanded,
      tagKeyList: filteredTagKeys,
      nestChildren: shouldNestChildrenAndFetchParents,
      referenceTime,
      runData,
      runUuidsMatchingFilter,
      runsPinned,
      runsHidden,
    });

    setVisibleRuns(runs);
  }, [
    runData,
    isLoadingRuns,
    experiments,
    metricKeyList,
    modelVersionsByRunUuid,
    paramKeyList,
    runsExpanded,
    filteredTagKeys,
    shouldNestChildrenAndFetchParents,
    referenceTime,
    runsPinned,
    runsHidden,
    runUuidsMatchingFilter,
    requestError,
  ]);

  const [notificationsFn, notificationContainer] = useLegacyNotification();
  const showFetchedRunsNotifications = useFetchedRunsNotification(notificationsFn);

  const loadMoreRunsCallback = useCallback(() => {
    if (moreRunsAvailable && !isLoadingRuns) {
      // Don't do this if we're loading runs
      // to prevent too many requests from being
      // sent out
      loadMoreRuns().then((runs) => {
        // Display notification about freshly loaded runs
        showFetchedRunsNotifications(runs, runInfos);
      });
    }
  }, [moreRunsAvailable, isLoadingRuns, loadMoreRuns, runInfos, showFetchedRunsNotifications]);

  useEffect(() => {
    onMaximizedChange?.(viewState.viewMaximized);
  }, [viewState.viewMaximized, onMaximizedChange]);

  const datasetSelected = useCallback((dataset: RunDatasetWithTags, run: RunRowType) => {
    setSelectedDatasetWithRun({ datasetWithTags: dataset, runData: run });
    setIsDrawerOpen(true);
  }, []);

  return (
    <CreateNewRunContextProvider visibleRuns={visibleRuns}>
      <ExperimentViewRunsControls
        viewState={viewState}
        updateViewState={updateViewState}
        runsData={runsData}
        searchFacetsState={searchFacetsState}
        updateSearchFacets={updateSearchFacets}
        experimentId={experiment_id}
        requestError={requestError}
        expandRows={expandRows}
        updateExpandRows={updateExpandRows}
      />
      <div css={styles.createRunsTableWrapper(isComparingRuns)}>
        <ExperimentViewRunsTable
          experiments={experiments}
          runsData={runsData}
          searchFacetsState={searchFacetsState}
          viewState={viewState}
          isLoading={isLoadingRuns}
          updateSearchFacets={updateSearchFacets}
          updateViewState={updateViewState}
          onAddColumnClicked={addColumnClicked}
          rowsData={visibleRuns}
          loadMoreRunsFunc={loadMoreRunsCallback}
          moreRunsAvailable={moreRunsAvailable}
          onDatasetSelected={datasetSelected}
          expandRows={expandRows}
        />
        {compareRunsMode === 'CHART' && (
          <RunsCompare
            isLoading={isLoadingRuns}
            comparedRuns={visibleRuns}
            metricKeyList={runsData.metricKeyList}
            paramKeyList={runsData.paramKeyList}
            experimentTags={runsData.experimentTags}
            searchFacetsState={searchFacetsState}
            updateSearchFacets={updateSearchFacets}
          />
        )}
        {compareRunsMode === 'ARTIFACT' && shouldEnableArtifactBasedEvaluation() && (
          <EvaluationArtifactCompareView
            comparedRuns={visibleRuns}
            viewState={viewState}
            updateViewState={updateViewState}
            updateSearchFacets={updateSearchFacets}
            onDatasetSelected={datasetSelected}
          />
        )}
        {notificationContainer}
        {selectedDatasetWithRun && (
          <ExperimentViewDatasetDrawer
            isOpen={isDrawerOpen}
            setIsOpen={setIsDrawerOpen}
            selectedDatasetWithRun={selectedDatasetWithRun}
            setSelectedDatasetWithRun={setSelectedDatasetWithRun}
          />
        )}
      </div>
    </CreateNewRunContextProvider>
  );
});

const styles = {
  createRunsTableWrapper: (isComparingRuns: boolean) => ({
    minHeight: 225, // This is the exact height for displaying a minimum five rows and table header
    height: '100%',
    display: 'grid',
    position: 'relative' as const,
    // When comparing runs, we fix the table width to 310px.
    // We can consider making it resizable by a user.
    gridTemplateColumns: isComparingRuns ? '310px 1fr' : '1fr',
  }),
  loadingFooter: () => ({
    justifyContent: 'center',
    alignItems: 'center',
    display: 'flex',
    height: '72px',
  }),
};

/**
 * Concrete actions for GetExperimentRuns context provider
 */
const getExperimentRunsActions = {
  searchRunsApi,
  loadMoreRunsApi,
  searchRunsPayload,
  searchModelVersionsApi,
};

/**
 * Function mapping redux state used in connect()
 */
const mapStateToProps = (
  state: { entities: ExperimentStoreEntities },
  params: ExperimentRunsSelectorParams,
) => {
  return { runsData: experimentRunsSelector(state, params) };
};

/**
 * Component responsible for displaying runs table with its set of
 * respective sort and filter controls on the experiment page.
 */
export const ExperimentViewRunsConnect: React.ComponentType<ExperimentViewRunsOwnProps> = connect(
  // mapStateToProps function:
  mapStateToProps,
  // mapDispatchToProps function (not provided):
  undefined,
  // mergeProps function (not provided):
  undefined,
  {
    // We're interested only in certain entities sub-tree so we won't
    // re-render on other state changes (e.g. API request IDs or metric history)
    areStatesEqual: (nextState, prevState) =>
      nextState.entities.experimentTagsByExperimentId ===
        prevState.entities.experimentTagsByExperimentId &&
      nextState.entities.latestMetricsByRunUuid === prevState.entities.latestMetricsByRunUuid &&
      nextState.entities.modelVersionsByRunUuid === prevState.entities.modelVersionsByRunUuid &&
      nextState.entities.paramsByRunUuid === prevState.entities.paramsByRunUuid &&
      nextState.entities.runInfosByUuid === prevState.entities.runInfosByUuid &&
      nextState.entities.runUuidsMatchingFilter === prevState.entities.runUuidsMatchingFilter &&
      nextState.entities.tagsByRunUuid === prevState.entities.tagsByRunUuid,
  },
)(ExperimentViewRunsImpl);

/**
 * This component serves as a layer for injecting props necessary for connect() to work
 */
export const ExperimentViewRunsInjectFilters = (props: ExperimentViewRunsOwnProps) => {
  const { searchFacetsState } = useFetchExperimentRuns();
  if (props.isLoading) {
    return <LegacySkeleton active />;
  }
  return (
    <ExperimentViewRunsConnect
      {...props}
      modelVersionFilter={searchFacetsState.modelVersionFilter as MODEL_VERSION_FILTER}
      lifecycleFilter={searchFacetsState.lifecycleFilter as LIFECYCLE_FILTER}
      datasetsFilter={searchFacetsState.datasetsFilter as DatasetSummary[]}
    />
  );
};

/**
 * This component serves as a layer for creating context for searching runs
 * and provides implementations of necessary redux actions.
 */
export const ExperimentViewRunsInjectContext = (props: ExperimentViewRunsOwnProps) => (
  <GetExperimentRunsContextProvider actions={getExperimentRunsActions}>
    <ExperimentViewRunsInjectFilters {...props} />
  </GetExperimentRunsContextProvider>
);

/**
 * Export context injection layer as a main entry point
 */
export const ExperimentViewRuns = React.memo(ExperimentViewRunsInjectContext);
