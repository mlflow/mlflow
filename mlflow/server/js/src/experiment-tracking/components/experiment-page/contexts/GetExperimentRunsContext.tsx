import { isFunction } from 'lodash';
import React, { createContext, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useHistory } from 'react-router-dom';
import RequestStateWrapper from '../../../../common/components/RequestStateWrapper';
import { loadMoreRunsApi, searchRunsApi, searchRunsPayload } from '../../../actions';
import { useExperimentIds } from '../hooks/useExperimentIds';
import { SearchExperimentRunsFacetsState } from '../models/SearchExperimentRunsFacetsState';
import Utils from '../../../../common/utils/Utils';
import { searchModelVersionsApi } from '../../../../model-registry/actions';
import { UpdateExperimentSearchFacetsFn } from '../../../types';
import { useAsyncDispatch } from '../hooks/useAsyncDispatch';
import {
  createSearchRunsParams,
  fetchModelVersionsForRuns,
  shouldRefetchRuns,
} from '../utils/experimentPage.fetch-utils';
import {
  persistExperimentSearchFacetsState,
  restoreExperimentSearchFacetsState,
} from '../utils/persistSearchFacets';

export interface GetExperimentRunsContextActions {
  searchRunsApi: typeof searchRunsApi;
  loadMoreRunsApi: typeof loadMoreRunsApi;
  searchRunsPayload: typeof searchRunsPayload;
  searchModelVersionsApi: typeof searchModelVersionsApi;
}

export interface GetExperimentRunsContextType {
  /**
   * Represents the currently used filter/sort model
   */
  searchFacetsState: SearchExperimentRunsFacetsState;

  /**
   * Indicates if runs are being loaded at the moment
   */
  isLoadingRuns: boolean;

  /**
   * Function used to (re)fetch runs with the currently used filter set.
   * Use scenarios: initial fetch, refreshing the list.
   */
  fetchExperimentRuns: () => void;

  /**
   * Function used to update the filter set and fetch new set of runs.
   * First parameter is the subset of fields that the current sort/filter model will be merged with.
   * If the second parameter is set to true, it will force re-fetching even if there
   * are no sufficient changes to the model.
   */
  updateSearchFacets: UpdateExperimentSearchFacetsFn;

  /**
   * Function used to load more runs (if available) using currently used filters
   */
  loadMoreRuns: () => void;

  /**
   * Contains error descriptor if fetching runs failed
   */
  requestError: any;

  /**
   * All run-related actions creators
   */
  actions: GetExperimentRunsContextActions;

  /**
   * True if there are more paginated runs available
   */
  moreRunsAvailable: boolean;

  /**
   * Returns true if user has not changed sort/filter model and is using the default one
   */
  isPristine: () => boolean;
}

/**
 * Wrapper context that serves two purposes:
 * - aggregates concrete redux actions necessary to perform experiments run search
 * - holds currently used state with sort and filter options, also is responsible for persisting it
 */
export const GetExperimentRunsContext = createContext<GetExperimentRunsContextType | null>(null);

/**
 * Provider component for SearchExperimentRunsContext.
 * Accepts concrete redux actions for searching runs.
 */
export const GetExperimentRunsContextProvider = ({
  children,
  actions,
}: React.PropsWithChildren<{
  actions: GetExperimentRunsContextActions;
}>) => {
  const history = useHistory();
  const experimentIds = useExperimentIds();
  const dispatch = useAsyncDispatch();

  const [searchRunsRequestId, setSearchRunsRequestId] = useState<string>('');
  const [isLoadingRuns, setIsLoadingRuns] = useState(false);
  const [moreRunsAvailable, setMoreRunsAvailable] = useState(false);
  const [requestError, setRequestError] = useState<any>(null);

  const experimentIdsHash = useMemo(() => JSON.stringify(experimentIds.sort()), [experimentIds]);

  const [searchFacetsState, setSearchFacetsState] = useState<SearchExperimentRunsFacetsState>(
    () => {
      // useState() initialization function that restores current search facets state
      const { queryString, state } = restoreExperimentSearchFacetsState(
        history.location.search,
        experimentIdsHash,
      );

      // If resulting query string differs from the current one, replace it.
      if (history.location.search !== queryString) {
        history.replace(`${history.location.pathname}${queryString}`);
      }
      return state;
    },
  );

  // Next page token is not a stateful field and can be mutable.
  const nextPageToken = useRef<string>('');

  // Indicates reference time for fetching subsuquent pages which
  // requires us to keep the same startTime parameter value.
  // Not a stateful field.
  const referenceTime = useRef<string>('');

  // Indicates if user has explicitly changed the sort/filter state
  const isPristineFlag = useRef(true);
  // Immutable method that returns value from the mutable flag
  const isPristine = useCallback(() => isPristineFlag.current, []);

  const internalFetchExperimentRuns = useCallback(
    (
      requestSearchFacetsState: SearchExperimentRunsFacetsState,
      requestExperimentIds: string[],
      requestReferenceTime = Date.now(),
      requestNextPageToken?: string,
    ) => {
      const loadMore = Boolean(requestNextPageToken);
      if (!loadMore) {
        referenceTime.current = requestReferenceTime;
      }

      const actionToUse = loadMore ? actions.loadMoreRunsApi : actions.searchRunsApi;

      const action = actionToUse(
        createSearchRunsParams(
          requestExperimentIds,
          requestSearchFacetsState,
          requestReferenceTime,
          requestNextPageToken || undefined,
        ),
      );

      // Immediately set loading runs flag, don't wait for RequestStateWrapper
      // otherwise it will result in the unnecessary rerender
      setIsLoadingRuns(true);
      dispatch(action)
        .then(({ value }) => {
          nextPageToken.current = value.next_page_token;
          setMoreRunsAvailable(Boolean(value.next_page_token));
          fetchModelVersionsForRuns(value.runs || [], actions.searchModelVersionsApi, dispatch);
        })
        .catch((e) => {
          Utils.logErrorAndNotifyUser(e);
        });

      setSearchRunsRequestId(action.meta.id);
    },
    [dispatch, actions],
  );

  const loadMoreRuns = useCallback(() => {
    internalFetchExperimentRuns(
      searchFacetsState,
      experimentIds,
      referenceTime.current || undefined,
      nextPageToken.current || undefined,
    );
  }, [internalFetchExperimentRuns, searchFacetsState, experimentIds]);

  /**
   * Fetches fresh batch of runs using current sort model
   */
  const fetchExperimentRuns = useCallback(() => {
    internalFetchExperimentRuns(searchFacetsState, experimentIds);
  }, [experimentIds, internalFetchExperimentRuns, searchFacetsState]);

  const persistState = useCallback(
    (sortFilterModelToSave: SearchExperimentRunsFacetsState) => {
      const newQueryString = persistExperimentSearchFacetsState(
        sortFilterModelToSave,
        experimentIdsHash,
        history.location.search,
      );
      if (history.location.search !== newQueryString) {
        history.push(`${history.location.pathname}${newQueryString}`);
      }
    },
    [history, experimentIdsHash],
  );

  /**
   * Updates current search facets model and invokes re-fetching runs if necessary.
   * Persists new search state in storage and URL.
   *
   * @param newFilterModel A search facets model to replace the current one. It can
   *                       be provided in the form of the partial object that will be merged
   *                       with the current one or the setter function having the current model
   *                       provided as a parameter and returning the merged model, similarly to the
   *                       React's `setState()` pattern.
   * @param updateOptions  Additional update options provided:
   *                       - `forceRefresh` - if set to `true`, will force re-fetching runs
   *                         regardless of the resulting search facets model.
   *                       - `preservePristine` - if set to `true`, will not change internal "pristine"
   *                         flag used to detect if user have tampered with the search model.
   *
   * @examples
   * ```
   * // Simple use, updates search filter
   * updateSearchFacets({
   *   searchFilter: 'attributes.start_time > 1000',
   * });
   *
   * // Setter function variant, updates selected columns
   * updateSearchFacets((currentModel) => ({
   *   ...currentModel,
   *   selectedColumns: [...currentModel.selectedColumns, 'someColumn'],
   * }));
   *
   * // Using function to force re-fetch while retaining pristine flag
   * const forceRefreshModel = updateSearchFacets({}, {
   *   forceRefresh: true,
   *   preservePristine: true,
   * });
   * ```
   */
  const updateSearchFacets = useCallback<UpdateExperimentSearchFacetsFn>(
    (newFilterModel, updateOptions = {}) => {
      const { forceRefresh = false, preservePristine = false } = updateOptions;
      // While dispatching new state, append new filter model
      // and fetch new runs using it
      setSearchFacetsState((oldModel) => {
        const newModel = isFunction(newFilterModel)
          ? newFilterModel(oldModel)
          : { ...oldModel, ...newFilterModel };
        if (forceRefresh || shouldRefetchRuns(oldModel, newModel)) {
          internalFetchExperimentRuns(newModel, experimentIds);
        }
        persistState(newModel);
        return newModel;
      });
      // Update the flag which indicates that the user have performed
      // a search with changed sort/filter model
      if (!preservePristine) {
        isPristineFlag.current = false;
      }
    },
    [experimentIds, internalFetchExperimentRuns, persistState],
  );

  /**
   * Dynamically restore searchFacets state on history navigation.
   * Note: MLFlow running in iFrame won't get proper history pop updates
   * from overrarching router, meaning that history.listen() won't work at all. In this case,
   * the page will get reloaded and useState()'s initialization function will restore the state instead.
   */
  useEffect(() => {
    return history.listen((location, action) => {
      if (action === 'POP') {
        const { state } = restoreExperimentSearchFacetsState(location.search, experimentIdsHash);
        updateSearchFacets(state);
      }
    });
  }, [history, experimentIdsHash, updateSearchFacets]);

  const contextValue = useMemo(
    () => ({
      actions,
      searchFacetsState,
      fetchExperimentRuns,
      updateSearchFacets,
      loadMoreRuns,
      requestError,
      isLoadingRuns,
      moreRunsAvailable,
      isPristine,
    }),
    [
      actions,
      searchFacetsState,
      fetchExperimentRuns,
      loadMoreRuns,
      requestError,
      isLoadingRuns,
      updateSearchFacets,
      moreRunsAvailable,
      isPristine,
    ],
  );

  const renderFn = (_isLoading: false, _renderError: any, requests: any[]) => {
    /**
     * TODO:
     * Defer setting this state because currently it might happen inside
     * RequestStateWrapper's render function which causes React to act up.
     * Either rebuild RequestStateWrapper or introduce some workaround.
     */
    setIsLoadingRuns(requests.some((r) => r.id === searchRunsRequestId && r.active));

    requests.forEach((request) => {
      if (request.error) {
        setRequestError(request.error);
      }
    });
    return children;
  };

  return (
    <GetExperimentRunsContext.Provider value={contextValue}>
      <RequestStateWrapper
        shouldOptimisticallyRender
        // eslint-disable-next-line no-trailing-spaces
        requestIds={searchRunsRequestId ? [searchRunsRequestId] : []}
      >
        {renderFn}
      </RequestStateWrapper>
    </GetExperimentRunsContext.Provider>
  );
};
