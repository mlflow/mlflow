import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { connect } from 'react-redux';
import {
  ExperimentEntity,
  ExperimentStoreEntities,
  LIFECYCLE_FILTER,
  MODEL_VERSION_FILTER,
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
import { ExperimentViewLoadMore } from './ExperimentViewLoadMore';
import Utils from '../../../../../common/utils/Utils';
import { ATTRIBUTE_COLUMN_SORT_KEY } from '../../../../constants';
import { RunRowType } from '../../utils/experimentPage.row-types';
import { prepareRunsGridData } from '../../utils/experimentPage.row-utils';
import { RunsCompare } from '../../../runs-compare/RunsCompare';

export interface ExperimentViewRunsOwnProps {
  experiments: ExperimentEntity[];
  modelVersionFilter?: MODEL_VERSION_FILTER;
  lifecycleFilter?: LIFECYCLE_FILTER;
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
  const { experiments, runsData } = props;

  // Persistable sort/filter model state is taken from the context
  const {
    searchFacetsState,
    updateSearchFacets,
    fetchExperimentRuns,
    isLoadingRuns,
    loadMoreRuns,
    moreRunsAvailable,
  } = useFetchExperimentRuns();

  const [visibleRuns, setVisibleRuns] = useState<RunRowType[]>([]);

  // Non-persistable view model state is being created locally
  const [viewState, setViewState] = useState(new SearchExperimentRunsViewState());

  // Initial fetch of runs after mounting
  useEffect(() => {
    fetchExperimentRuns();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const updateViewState = useCallback<UpdateExperimentViewStateFn>(
    (newPartialViewState) =>
      setViewState((currentViewState) => ({ ...currentViewState, ...newPartialViewState })),
    [],
  );

  const addColumnClicked = useCallback(() => {
    updateViewState({ columnSelectorVisible: true });
  }, [updateViewState]);

  const { orderByKey, searchFilter, runsExpanded, runsPinned, isComparingRuns } = searchFacetsState;

  const shouldNestChildrenAndFetchParents = useMemo(
    () => (!orderByKey && !searchFilter) || orderByKey === ATTRIBUTE_COLUMN_SORT_KEY.DATE,
    [orderByKey, searchFilter],
  );

  // Value used a reference for the "date" column
  const [referenceTime, setReferenceTime] = useState(createCurrentTime);

  const {
    paramKeyList,
    metricKeyList,
    tagsList,
    modelVersionsByRunUuid,
    paramsList,
    metricsList,
    runInfos,
    runUuidsMatchingFilter,
  } = runsData;

  // We're setting new reference date only when new runs data package has arrived
  useEffect(() => {
    setReferenceTime(createCurrentTime);
  }, [runInfos]);

  const filteredTagKeys = useMemo(() => Utils.getVisibleTagKeyList(tagsList), [tagsList]);

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
      runData: runInfos.map((runInfo, index) => ({
        runInfo,
        params: paramsList[index],
        metrics: metricsList[index],
        tags: tagsList[index],
      })),
      runUuidsMatchingFilter,
      runsPinned,
    });

    setVisibleRuns(runs);
  }, [
    isLoadingRuns,
    experiments,
    metricKeyList,
    metricsList,
    modelVersionsByRunUuid,
    paramKeyList,
    paramsList,
    runInfos,
    runsExpanded,
    tagsList,
    filteredTagKeys,
    shouldNestChildrenAndFetchParents,
    referenceTime,
    runsPinned,
    runUuidsMatchingFilter,
  ]);

  return (
    <>
      <ExperimentViewRunsControls
        viewState={viewState}
        updateViewState={updateViewState}
        runsData={runsData}
        searchFacetsState={searchFacetsState}
        updateSearchFacets={updateSearchFacets}
        visibleRowsCount={visibleRuns.length}
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
        />
        {isComparingRuns && (
          <RunsCompare
            isLoading={isLoadingRuns}
            comparedRuns={visibleRuns}
            metricKeyList={runsData.metricKeyList}
            paramKeyList={runsData.paramKeyList}
            searchFacetsState={searchFacetsState}
            updateSearchFacets={updateSearchFacets}
          />
        )}
      </div>
      <ExperimentViewLoadMore
        isLoadingRuns={isLoadingRuns}
        loadMoreRuns={loadMoreRuns}
        moreRunsAvailable={moreRunsAvailable}
      />
    </>
  );
});

const styles = {
  createRunsTableWrapper: (isComparingRuns: boolean) => ({
    display: 'grid',
    // When comparing runs, we fix the table width to 260px.
    // We can consider making it resizable by a user.
    gridTemplateColumns: isComparingRuns ? '260px 1fr' : '1fr',
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
    // We're interested only in "entities" sub-tree so we won't
    // re-render on other state changes (e.g. API request IDs)
    areStatesEqual: (nextState, prevState) => nextState.entities === prevState.entities,
  },
)(ExperimentViewRunsImpl);

/**
 * This component serves as a layer for injecting props necessary for connect() to work
 */
export const ExperimentViewRunsInjectFilters = (props: ExperimentViewRunsOwnProps) => {
  const { searchFacetsState } = useFetchExperimentRuns();
  return (
    <ExperimentViewRunsConnect
      {...props}
      modelVersionFilter={searchFacetsState.modelVersionFilter as MODEL_VERSION_FILTER}
      lifecycleFilter={searchFacetsState.lifecycleFilter as LIFECYCLE_FILTER}
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
export const ExperimentViewRuns = ExperimentViewRunsInjectContext;
