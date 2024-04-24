import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { useDispatch } from 'react-redux';
import { ReduxState, ThunkDispatch } from '../../../../redux-types';
import { loadMoreRunsApi, searchRunsApi } from '../../../actions';
import { ExperimentPageUIStateV2 } from '../models/ExperimentPageUIStateV2';
import { createSearchRunsParams, fetchModelVersionsForRuns } from '../utils/experimentPage.fetch-utils';
import { ExperimentRunsSelectorResult, experimentRunsSelector } from '../utils/experimentRuns.selector';
import { ExperimentQueryParamsSearchFacets } from './useExperimentPageSearchFacets';
import { ExperimentPageSearchFacetsStateV2 } from '../models/ExperimentPageSearchFacetsStateV2';
import { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';
import { searchModelVersionsApi } from '../../../../model-registry/actions';
import Utils from '../../../../common/utils/Utils';

type RunsRequestParams = ReturnType<typeof createSearchRunsParams> & {
  requestedFacets: ExperimentPageSearchFacetsStateV2;
};

/**
 * This hook will request for new runs data based on the current experiment page search facets and UI state.
 * Replaces GetExperimentRunsContext and a substantial portion of <ExperimentRuns> component stack.
 */
export const useExperimentRuns = (
  uiState: ExperimentPageUIStateV2,
  searchFacets: ExperimentQueryParamsSearchFacets | null,
  experimentIds: string[],
  disabled = false,
) => {
  const dispatch = useDispatch<ThunkDispatch>();

  const [runsData, setRunsData] = useState<ExperimentRunsSelectorResult>(() => createEmptyRunsResult());

  const persistKey = useMemo(() => (experimentIds ? JSON.stringify(experimentIds.sort()) : null), [experimentIds]);
  const [isLoadingRuns, setIsLoadingRuns] = useState(true);
  const [isInitialLoadingRuns, setIsInitialLoadingRuns] = useState(true);
  const [nextPageToken, setNextPageToken] = useState<string | null>(null);
  const [requestError, setRequestError] = useState<ErrorWrapper | null>(null);
  const cachedPinnedRuns = useRef<string[]>([]);

  // Reset initial loading state when experiment IDs change
  useEffect(() => {
    if (disabled) {
      return;
    }
    setIsInitialLoadingRuns(true);
    setRunsData(createEmptyRunsResult());
  }, [persistKey, disabled]);

  const setResultRunsData = useCallback(
    (store: ReduxState, experimentIds: string[], requestedFacets: ExperimentPageSearchFacetsStateV2) => {
      setRunsData(
        experimentRunsSelector(store, {
          datasetsFilter: requestedFacets.datasetsFilter,
          lifecycleFilter: requestedFacets.lifecycleFilter,
          modelVersionFilter: requestedFacets.modelVersionFilter,
          // In the new version of the view state, experiment IDs are used instead of full experiment entities:
          experiments: [],
          experimentIds,
        }),
      );
    },
    [],
  );

  useEffect(() => {
    cachedPinnedRuns.current = uiState.runsPinned;
  }, [uiState.runsPinned]);

  // Calculate actual params to use for fetching runs
  const createFetchRunsRequestParams = useCallback(
    (searchFacets: ExperimentQueryParamsSearchFacets | null, experimentIds: string[]): RunsRequestParams | null => {
      if (!searchFacets || !experimentIds.length) {
        return null;
      }
      const searchParams = createSearchRunsParams(
        experimentIds,
        // We use the cached pinned runs here, but not in the hook deps - we want to use it, but not trigger a new request
        { ...searchFacets, runsPinned: cachedPinnedRuns.current },
        Date.now(),
      );
      return { ...searchParams, requestedFacets: searchFacets };
    },
    [],
  );

  const loadModelVersions = useCallback(
    (runs: Parameters<typeof fetchModelVersionsForRuns>[0]) => {
      fetchModelVersionsForRuns(runs || [], searchModelVersionsApi, dispatch);
    },
    [dispatch],
  );

  // Main function for fetching runs
  const fetchRuns = useCallback(
    (params: RunsRequestParams) =>
      dispatch((thunkDispatch: ThunkDispatch, getStore: () => ReduxState) => {
        setIsLoadingRuns(true);
        return thunkDispatch((params.pageToken ? loadMoreRunsApi : searchRunsApi)(params))
          .then(async ({ value }) => {
            setNextPageToken(value.next_page_token || null);
            setIsLoadingRuns(false);
            setIsInitialLoadingRuns(false);

            // We rely on redux reducer to update the state with new runs data,
            // then we pick it up from the store. This benefits other pages that use same data
            // from the same store slice (e.g. run details page). Will be changed when moving to graphQL.
            setResultRunsData(getStore(), params.experimentIds, params.requestedFacets);

            // In the end, load model versions for the fetched runs
            loadModelVersions(value.runs || []);
            return value;
          })
          .catch((e) => {
            setIsLoadingRuns(false);
            setIsInitialLoadingRuns(false);
            setRequestError(e);
            Utils.logErrorAndNotifyUser(e);
          });
      }),
    [dispatch, setResultRunsData, loadModelVersions],
  );

  // Fetch runs when new request params are available
  // (e.g. after search facets change)
  useEffect(() => {
    if (disabled) {
      return;
    }
    const requestParams = createFetchRunsRequestParams(searchFacets, experimentIds);
    if (requestParams) {
      fetchRuns(requestParams);
    }
  }, [createFetchRunsRequestParams, fetchRuns, dispatch, disabled, searchFacets, experimentIds]);

  const loadMoreRuns = async () => {
    const requestParams = createFetchRunsRequestParams(searchFacets, experimentIds);
    if (!nextPageToken || !requestParams) {
      return [];
    }
    return fetchRuns({ ...requestParams, pageToken: nextPageToken });
  };

  const refreshRuns = () => {
    const requestParams = createFetchRunsRequestParams(searchFacets, experimentIds);
    if (requestParams) {
      fetchRuns(requestParams);
    }
  };

  return {
    isLoadingRuns,
    moreRunsAvailable: Boolean(nextPageToken),
    refreshRuns,
    loadMoreRuns,
    isInitialLoadingRuns,
    runsData,
    requestError,
  };
};

const createEmptyRunsResult = () => ({
  datasetsList: [],
  experimentTags: {},
  metricKeyList: [],
  metricsList: [],
  modelVersionsByRunUuid: {},
  paramKeyList: [],
  paramsList: [],
  runInfos: [],
  runUuidsMatchingFilter: [],
  tagsList: [],
});
