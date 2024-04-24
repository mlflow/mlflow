import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { connect, useSelector } from 'react-redux';
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
import { ATTRIBUTE_COLUMN_SORT_KEY } from '../../../../constants';
import { RunRowType } from '../../utils/experimentPage.row-types';
import { prepareRunsGridData, useExperimentRunRows } from '../../utils/experimentPage.row-utils';
import { RunsCompare } from '../../../runs-compare/RunsCompare';
import { useFetchedRunsNotification } from '../../hooks/useFetchedRunsNotification';
import { DatasetWithRunType, ExperimentViewDatasetDrawer } from './ExperimentViewDatasetDrawer';
import { useExperimentViewLocalStore } from '../../hooks/useExperimentViewLocalStore';
import { useAutoExpandRunRows } from '../../hooks/useAutoExpandRunRows';
import { EvaluationArtifactCompareView } from '../../../evaluation-artifacts-compare/EvaluationArtifactCompareView';
import {
  shouldEnableMetricChartsGrouping,
  shouldEnableShareExperimentViewByTags,
} from '../../../../../common/utils/FeatureUtils';
import { CreateNewRunContextProvider } from '../../hooks/useCreateNewRun';
import { useExperimentPageViewMode } from '../../hooks/useExperimentPageViewMode';
import { SearchExperimentRunsFacetsState } from '../../models/SearchExperimentRunsFacetsState';
import { ExperimentPageUIStateV2 } from '../../models/ExperimentPageUIStateV2';
import { noop } from 'lodash';
import { RunsCompareV2 } from '../../../runs-compare/RunsCompareV2';
import { ErrorWrapper } from '../../../../../common/utils/ErrorWrapper';
import { ReduxState } from '../../../../../redux-types';

export interface ExperimentViewRunsOwnProps {
  isLoading: boolean;
  experiments: ExperimentEntity[];
  modelVersionFilter?: MODEL_VERSION_FILTER;
  lifecycleFilter?: LIFECYCLE_FILTER;
  datasetsFilter?: DatasetSummary[];
  onMaximizedChange?: (newIsMaximized: boolean) => void;

  searchFacetsState: SearchExperimentRunsFacetsState;
  uiState: ExperimentPageUIStateV2;
}

// In the new view state model, those props are provided directly to the runs component
export interface ExperimentViewRunsV2Props {
  isLoadingRuns: boolean;
  loadMoreRuns: () => Promise<any>;
  moreRunsAvailable: boolean;
  isPristine: () => boolean;
  requestError: ErrorWrapper | null;
  refreshRuns: () => void;
}

export interface ExperimentViewRunsProps extends ExperimentViewRunsOwnProps, ExperimentViewRunsV2Props {
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

/**
 * An internal hook for selecting proper version of functions for
 * loading runs and manipulating view state. This is necessary for
 * backward compatibility with the old view state model.
 *
 * If the feature flag is disabled, the old context-based view state model is used.
 */
const useCompatRunsViewState = (props: ExperimentViewRunsProps) => {
  const usingNewViewStateModel = shouldEnableShareExperimentViewByTags();
  const { searchFacetsState: searchFacetsStateFromProps, experiments, onMaximizedChange, uiState } = props;
  const {
    searchFacetsState: searchFacetsStateFromContext,
    updateSearchFacets: updateSearchFacetsFromContext,
    refreshRuns: refreshRunsFromContext,
    isLoadingRuns: isLoadingRunsFromContext,
    loadMoreRuns: loadMoreRunsFromContext,
    moreRunsAvailable: moreRunsAvailableFromContext,
    isPristine,
    requestError: requestErrorFromContext,
  } = useFetchExperimentRuns();

  // In new view state model, get values from the props instead of the context.
  const searchFacetsState = usingNewViewStateModel ? searchFacetsStateFromProps : searchFacetsStateFromContext;
  const updateSearchFacets = usingNewViewStateModel ? noop : updateSearchFacetsFromContext;
  const refreshRuns = usingNewViewStateModel ? props.refreshRuns : refreshRunsFromContext;
  const isLoadingRuns = usingNewViewStateModel ? props.isLoadingRuns : isLoadingRunsFromContext;
  const loadMoreRuns = usingNewViewStateModel ? props.loadMoreRuns : loadMoreRunsFromContext;
  const moreRunsAvailable = usingNewViewStateModel ? props.moreRunsAvailable : moreRunsAvailableFromContext;
  const requestError = usingNewViewStateModel ? props.requestError : requestErrorFromContext;

  return {
    searchFacetsState,
    updateSearchFacets,
    isLoadingRuns,
    loadMoreRuns,
    moreRunsAvailable,
    isPristine,
    requestError,
    refreshRuns,
    experiments,
    onMaximizedChange,
    uiState,
  };
};

export const ExperimentViewRunsImpl = React.memo((props: ExperimentViewRunsProps) => {
  const usingNewViewStateModel = shouldEnableShareExperimentViewByTags();
  const [compareRunsModeFromUrl] = useExperimentPageViewMode();
  const { experiments, runsData, onMaximizedChange, uiState } = props;

  // Persistable sort/filter model state is taken from the context
  const {
    searchFacetsState,
    updateSearchFacets,
    isLoadingRuns,
    loadMoreRuns,
    moreRunsAvailable,
    isPristine,
    requestError,
    refreshRuns,
  } = useCompatRunsViewState(props);

  const [visibleRunsLegacy, setVisibleRunsLegacy] = useState<RunRowType[]>([]);

  // Non-persistable view model state is being created locally
  // TODO: Remove when fully migrated to new view state model (uiState)
  const [viewState, setViewState] = useState(new SearchExperimentRunsViewState());
  const runListHidden = usingNewViewStateModel ? uiState.runListHidden : viewState.runListHidden;

  const { experiment_id } = experiments[0];
  const expandRowsStore = useExperimentViewLocalStore(experiment_id);
  const [expandRows, updateExpandRows] = useState<boolean>(expandRowsStore.getItem('expandRows') === 'true');

  useEffect(() => {
    expandRowsStore.setItem('expandRows', expandRows);
  }, [expandRows, expandRowsStore]);

  const {
    paramKeyList,
    metricKeyList,
    tagsList,
    modelVersionsByRunUuid: modelVersionsByRunUuidFromRunsData,
    paramsList,
    metricsList,
    runInfos,
    runUuidsMatchingFilter,
    datasetsList,
  } = runsData;

  const modelVersionsByRunUuidFromStore = useSelector(({ entities }: ReduxState) => entities.modelVersionsByRunUuid);

  const modelVersionsByRunUuid = usingNewViewStateModel
    ? modelVersionsByRunUuidFromStore
    : modelVersionsByRunUuidFromRunsData;

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

  const { orderByKey, searchFilter } = searchFacetsState;
  // In new view state model, runs state is in the uiState instead of the searchFacetsState.
  const { runsPinned, runsExpanded, runsHidden } = usingNewViewStateModel ? uiState : searchFacetsState;

  // Use modernized view mode value getter if flag is set
  const compareRunsMode = usingNewViewStateModel ? compareRunsModeFromUrl : searchFacetsState.compareRunsMode;

  const isComparingRuns = compareRunsMode !== undefined;

  const updateViewState = useCallback<UpdateExperimentViewStateFn>(
    (newPartialViewState) => setViewState((currentViewState) => ({ ...currentViewState, ...newPartialViewState })),
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

  // Use new, memoized version of the row creation function.
  // Internally disabled if the flag is not set.
  const visibleRunsMemoized = useExperimentRunRows({
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
    groupBy: uiState.groupBy,
    groupsExpanded: uiState.groupsExpanded,
    runsHiddenMode: uiState.runsHiddenMode,
  });

  useEffect(() => {
    if (usingNewViewStateModel || isLoadingRuns) {
      return;
    }
    const rows = prepareRunsGridData({
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
      groupBy: '',
      groupsExpanded: {},
    });

    setVisibleRunsLegacy(rows);
  }, [
    usingNewViewStateModel,
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

  const visibleRuns = usingNewViewStateModel ? visibleRunsMemoized : visibleRunsLegacy;

  // TODO: Remove all hooks below after migration to new view state model
  // Automatically expand parent runs if necessary
  useAutoExpandRunRows(runData, visibleRuns, isPristine, updateSearchFacets, runsExpanded);

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
    <CreateNewRunContextProvider visibleRuns={visibleRuns} refreshRuns={refreshRuns}>
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
        refreshRuns={refreshRuns}
        uiState={uiState}
        isLoading={isLoadingRuns}
      />
      <div
        css={{
          minHeight: 225, // This is the exact height for displaying a minimum five rows and table header
          height: '100%',
          display: 'grid',
          position: 'relative' as const,
          gridTemplateColumns: isComparingRuns ? (runListHidden ? '10px 1fr' : '295px 1fr') : '1fr',
        }}
      >
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
          uiState={uiState}
        />
        {compareRunsMode === 'CHART' &&
          (shouldEnableMetricChartsGrouping() ? (
            <RunsCompareV2
              isLoading={isLoadingRuns}
              comparedRuns={visibleRuns}
              metricKeyList={runsData.metricKeyList}
              paramKeyList={runsData.paramKeyList}
              experimentTags={runsData.experimentTags}
              compareRunCharts={uiState.compareRunCharts}
              compareRunSections={uiState.compareRunSections}
              groupBy={uiState.groupBy}
            />
          ) : (
            <RunsCompare
              isLoading={isLoadingRuns}
              comparedRuns={visibleRuns}
              metricKeyList={runsData.metricKeyList}
              paramKeyList={runsData.paramKeyList}
              experimentTags={runsData.experimentTags}
              updateSearchFacets={updateSearchFacets}
              compareRunCharts={usingNewViewStateModel ? uiState.compareRunCharts : searchFacetsState.compareRunCharts}
            />
          ))}
        {compareRunsMode === 'ARTIFACT' && (
          <EvaluationArtifactCompareView
            comparedRuns={visibleRuns}
            viewState={viewState}
            updateViewState={updateViewState}
            updateSearchFacets={updateSearchFacets}
            onDatasetSelected={datasetSelected}
            disabled={Boolean(uiState.groupBy)}
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
const mapStateToProps = (state: { entities: ExperimentStoreEntities }, params: ExperimentRunsSelectorParams) => {
  return { runsData: experimentRunsSelector(state, params) };
};

/**
 * Component responsible for displaying runs table with its set of
 * respective sort and filter controls on the experiment page.
 * TODO(ML-35962): Disable this layer after migration to new view state model
 */
export const ExperimentViewRunsConnect: React.ComponentType<ExperimentViewRunsProps> = connect(
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
      nextState.entities.experimentTagsByExperimentId === prevState.entities.experimentTagsByExperimentId &&
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
 * TODO(ML-35962): Disable this layer after migration to new view state model
 */
export const ExperimentViewRunsInjectFilters = (props: ExperimentViewRunsProps) => {
  const { searchFacetsState: searchFacetsStateFromContext } = useFetchExperimentRuns();
  const { searchFacetsState: searchFacetsStateFromProps } = props;

  const searchFacetsState = shouldEnableShareExperimentViewByTags()
    ? searchFacetsStateFromProps
    : searchFacetsStateFromContext;

  if (props.isLoading) {
    return <LegacySkeleton active />;
  }
  return (
    <ExperimentViewRunsConnect
      {...props}
      modelVersionFilter={searchFacetsState.modelVersionFilter}
      lifecycleFilter={searchFacetsState.lifecycleFilter}
      datasetsFilter={searchFacetsState.datasetsFilter}
    />
  );
};

/**
 * This component serves as a layer for creating context for searching runs
 * and provides implementations of necessary redux actions.
 * TODO(ML-35962): Disable this layer after migration to new view state model
 */
export const ExperimentViewRunsInjectContext = (props: ExperimentViewRunsProps) => {
  return (
    <GetExperimentRunsContextProvider actions={getExperimentRunsActions}>
      <ExperimentViewRunsInjectFilters {...props} />
    </GetExperimentRunsContextProvider>
  );
};

/**
 * Export context injection layer as a main entry point
 * TODO(ML-35962): Disable this layer after migration to new view state model
 */
export const ExperimentViewRuns = React.memo(
  ExperimentViewRunsInjectContext as React.FC<Omit<ExperimentViewRunsOwnProps, keyof ExperimentViewRunsV2Props>>,
);
