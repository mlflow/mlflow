import React, { useCallback, useEffect, useState } from 'react';
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

export interface ExperimentViewRunsOwnProps {
  experiments: ExperimentEntity[];
  modelVersionFilter?: MODEL_VERSION_FILTER;
  lifecycleFilter?: LIFECYCLE_FILTER;
}

export interface ExperimentViewRunsProps extends ExperimentViewRunsOwnProps {
  runsData: ExperimentRunsSelectorResult;
}

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

  return (
    <>
      <ExperimentViewRunsControls
        viewState={viewState}
        updateViewState={updateViewState}
        runsData={runsData}
        searchFacetsState={searchFacetsState}
        updateSearchFacets={updateSearchFacets}
      />
      <ExperimentViewRunsTable
        experiments={experiments}
        runsData={runsData}
        searchFacetsState={searchFacetsState}
        viewState={viewState}
        isLoading={isLoadingRuns}
        updateSearchFacets={updateSearchFacets}
        updateViewState={updateViewState}
        onAddColumnClicked={addColumnClicked}
      />
      <ExperimentViewLoadMore
        isLoadingRuns={isLoadingRuns}
        loadMoreRuns={loadMoreRuns}
        moreRunsAvailable={moreRunsAvailable}
      />
    </>
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
