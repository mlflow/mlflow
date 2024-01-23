import { renderHook, act, type RenderHookResult } from '@testing-library/react-for-react-18';
import {
  ExperimentPageSearchFacetsStateV2,
  createExperimentPageSearchFacetsStateV2,
} from '../models/ExperimentPageSearchFacetsStateV2';
import { ExperimentPageUIStateV2, createExperimentPageUIStateV2 } from '../models/ExperimentPageUIStateV2';
import { useExperimentRuns } from './useExperimentRuns';
import { loadMoreRunsApi, searchRunsApi } from '../../../actions';
import { applyMiddleware, combineReducers, createStore } from 'redux';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import {
  experimentTagsByExperimentId,
  experimentsById,
  paramsByRunUuid,
  runDatasetsByUuid,
  runInfosByUuid,
  runUuidsMatchingFilter,
  tagsByRunUuid,
} from '../../../reducers/Reducers';
import { Provider } from 'react-redux';
import { latestMetricsByRunUuid, metricsByRunUuid } from '../../../reducers/MetricReducer';
import { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';
import Utils from '../../../../common/utils/Utils';
import { ExperimentRunsSelectorResult } from '../utils/experimentRuns.selector';

jest.mock('../../../actions', () => ({
  ...jest.requireActual('../../../actions'),
  searchRunsApi: jest.fn(),
  loadMoreRunsApi: jest.fn(),
}));

const MOCK_RESPONSE_DELAY = 1000;

const createInitialRunsResponse = () =>
  new Promise((resolve) =>
    setTimeout(
      () =>
        resolve({
          next_page_token: 'some_page_token',
          runs: [
            {
              info: {
                run_uuid: 'run_1',
                experiment_id: 'test-experiment',
                run_name: 'run_1_name',
                lifecycle_stage: 'active',
              },
              data: {
                metrics: [{ key: 'm1', value: 100, timestamp: 100, step: 1 }],
                params: [{ key: 'p1', value: 'p1_value' }],
                tags: [{ key: 't1', value: 't1_value' }],
              },
            },
            {
              info: {
                run_uuid: 'run_2',
                experiment_id: 'test-experiment',
                run_name: 'run_2_name',
                lifecycle_stage: 'active',
              },
              data: {
                metrics: [{ key: 'm2', value: 200, timestamp: 100, step: 1 }],
                tags: [],
              },
            },
            {
              info: {
                run_uuid: 'run_3',
                experiment_id: 'test-experiment',
                run_name: 'run_3_name',
                lifecycle_stage: 'deleted',
              },
              data: {
                tags: [],
              },
            },
          ],
        }),
      MOCK_RESPONSE_DELAY,
    ),
  );

const createLoadMoreRunsResponse = () =>
  new Promise((resolve) =>
    setTimeout(
      () =>
        resolve({
          runs: [
            {
              info: {
                run_uuid: 'run_4',
                experiment_id: 'test-experiment',
                run_name: 'run_4_name',
                lifecycle_stage: 'active',
              },
              data: {
                metrics: [{ key: 'm4', value: 400, timestamp: 100, step: 1 }],
                params: [{ key: 'p4', value: 'p4_value' }],
                tags: [],
              },
            },
          ],
        }),
      MOCK_RESPONSE_DELAY * 2,
    ),
  );

// For integration testing, use actual redux store with reducers
const store = createStore(
  combineReducers({
    entities: combineReducers({
      runInfosByUuid,
      experimentTagsByExperimentId,
      experimentsById,
      runDatasetsByUuid,
      latestMetricsByRunUuid,
      metricsByRunUuid,
      paramsByRunUuid,
      tagsByRunUuid,
      runUuidsMatchingFilter,
    }),
  }),
  applyMiddleware(thunk, promiseMiddleware()),
);

const testExperimentIds = ['test-experiment'];
const testUiState = createExperimentPageUIStateV2();
const testSearchFacets = createExperimentPageSearchFacetsStateV2();

// This suite tests useExperimentRuns hook, related reducers, actions and selectors.
describe('useExperimentRuns - integration test', () => {
  beforeAll(() => {
    jest.useFakeTimers();
  });
  afterAll(() => {
    jest.useRealTimers();
  });
  beforeEach(() => {
    jest
      .mocked(searchRunsApi)
      .mockReturnValue({ type: 'SEARCH_RUNS_API', payload: Promise.resolve({}), meta: { id: 0 } });
    jest
      .mocked(loadMoreRunsApi)
      .mockReturnValue({ type: 'LOAD_MORE_RUNS_API', payload: Promise.resolve({}), meta: { id: 0 } });
  });
  const renderTestHook = async (
    initialUiState = testUiState,
    initialSearchFacets = testSearchFacets,
    initialExperimentIds = testExperimentIds,
  ): Promise<RenderHookResult<ReturnType<typeof useExperimentRuns>, any>> => {
    let result: any;
    await act(async () => {
      result = renderHook(
        ({
          experimentIds,
          searchFacets,
          uiState,
        }: {
          uiState: ExperimentPageUIStateV2;
          searchFacets: ExperimentPageSearchFacetsStateV2;
          experimentIds: string[];
        }) => useExperimentRuns(uiState, searchFacets, experimentIds),
        {
          wrapper: ({ children }) => <Provider store={store}>{children}</Provider>,
          initialProps: {
            uiState: initialUiState,
            searchFacets: initialSearchFacets,
            experimentIds: initialExperimentIds,
          },
        },
      );
    });
    return result;
  };
  test('should call for new runs and return processed runs data', async () => {
    // Mock two responses - one for initial runs, one for load more runs
    jest.mocked(searchRunsApi).mockReturnValue({
      type: 'SEARCH_RUNS_API',
      payload: createInitialRunsResponse(),
      meta: { id: 0 },
    });
    jest.mocked(loadMoreRunsApi).mockReturnValue({
      type: 'LOAD_MORE_RUNS_API',
      payload: createLoadMoreRunsResponse(),
      meta: { id: 0 },
    });

    // Run the hook
    const { result, rerender } = await renderTestHook();

    // Check initial loading state
    expect(result.current.isLoadingRuns).toEqual(true);
    expect(result.current.isInitialLoadingRuns).toEqual(true);

    // Check that the search runs API was called with the expected params
    expect(searchRunsApi).toHaveBeenCalledTimes(1);
    expect(searchRunsApi).toHaveBeenLastCalledWith(
      expect.objectContaining({
        experimentIds: ['test-experiment'],
        filter: undefined,
        orderBy: ['attributes.start_time DESC'],
        pageToken: undefined,
        runViewType: 'ACTIVE_ONLY',
        runsPinned: [],
        shouldFetchParents: true,
      }),
    );

    // Advance timers to trigger the first response
    await act(async () => {
      jest.advanceTimersByTime(1500);
    });

    // Check that the hook is no longer loading
    expect(result.current.isLoadingRuns).toEqual(false);
    expect(result.current.isInitialLoadingRuns).toEqual(false);

    // Check that the runs data is as expected (one run is deleted)
    expect(result.current.runsData?.runInfos).toHaveLength(2);

    // Check that the metrics data corresponds to the payload
    expect(result.current.runsData?.metricsList).toEqual([
      [expect.objectContaining({ key: 'm1', value: 100 })],
      [expect.objectContaining({ key: 'm2', value: 200 })],
    ]);

    // Check that the params data is as expected
    expect(result.current.runsData?.paramsList).toEqual([
      [expect.objectContaining({ key: 'p1', value: 'p1_value' })],
      [],
    ]);

    // Check that the metric and param keys are as expected
    expect(result.current.runsData?.metricKeyList).toEqual(['m1', 'm2']);
    expect(result.current.runsData?.paramKeyList).toEqual(['p1']);

    // Check that the next page token is set
    expect(result.current.moreRunsAvailable).toEqual(true);

    // Change the search facets to include a filter
    const searchFacetsWithFilter = { ...testSearchFacets, searchFilter: 'metrics.m1 > 5' };
    await act(async () => {
      rerender({
        uiState: testUiState,
        searchFacets: searchFacetsWithFilter,
        experimentIds: testExperimentIds,
      });
    });

    // Check that the search runs API was called again with the expected params
    expect(searchRunsApi).toHaveBeenCalledTimes(2);
    expect(searchRunsApi).toHaveBeenLastCalledWith(
      expect.objectContaining({
        experimentIds: ['test-experiment'],
        filter: 'metrics.m1 > 5',
        orderBy: ['attributes.start_time DESC'],
        pageToken: undefined,
        runViewType: 'ACTIVE_ONLY',
        runsPinned: [],
        shouldFetchParents: true,
      }),
    );

    // Create a new UI state with a new set of pinned runs and rerender the hook
    const uiStateWithPinnedRuns = { ...testUiState, runsPinned: ['run_2'] };
    await act(async () => {
      rerender({
        uiState: uiStateWithPinnedRuns,
        searchFacets: searchFacetsWithFilter,
        experimentIds: testExperimentIds,
      });
    });

    // Check that the search runs API was not called again just due to new pinned runs
    expect(searchRunsApi).toHaveBeenCalledTimes(2);

    // Change the sort order and rerender the hook
    const searchFacetsWithSortByName = { ...testSearchFacets, orderByKey: 'tags.`mlflow.runName`' };
    await act(async () => {
      rerender({
        uiState: uiStateWithPinnedRuns,
        searchFacets: searchFacetsWithSortByName,
        experimentIds: testExperimentIds,
      });
    });

    // Check that the search runs API was called again with the both pinned runs and sort order
    expect(searchRunsApi).toHaveBeenCalledTimes(3);
    expect(searchRunsApi).toHaveBeenLastCalledWith(
      expect.objectContaining({
        experimentIds: ['test-experiment'],
        filter: undefined,
        orderBy: ['tags.`mlflow.runName` DESC'],
        pageToken: undefined,
        runViewType: 'ACTIVE_ONLY',
        runsPinned: ['run_2'],
        // Not sorted by creation date -> hierarchy is disabled -> no need to fetch parents
        shouldFetchParents: false,
      }),
    );

    // Invoke load more runs
    await act(async () => {
      result.current.loadMoreRuns();
    });

    // Check that the hook is loading
    expect(result.current.isLoadingRuns).toEqual(true);
    expect(result.current.isInitialLoadingRuns).toEqual(false);

    // Advance timers to trigger the second response
    await act(async () => {
      jest.advanceTimersByTime(1500);
    });

    // Check that the hook is no longer loading
    expect(result.current.isLoadingRuns).toEqual(false);
    expect(result.current.isInitialLoadingRuns).toEqual(false);

    // Check that the runs data is as expected (one run is added)
    expect(result.current.runsData?.runInfos).toHaveLength(3);

    // Check that the metrics data corresponds to the payload
    expect(result.current.runsData?.metricsList).toEqual([
      [expect.objectContaining({ key: 'm1', value: 100 })],
      [expect.objectContaining({ key: 'm2', value: 200 })],
      [expect.objectContaining({ key: 'm4', value: 400 })],
    ]);

    // Check that the params data is as expected
    expect(result.current.runsData?.paramsList).toEqual([
      [expect.objectContaining({ key: 'p1', value: 'p1_value' })],
      [],
      [expect.objectContaining({ key: 'p4', value: 'p4_value' })],
    ]);

    // Check that the metric and param key list contain new entries
    expect(result.current.runsData?.metricKeyList).toEqual(['m1', 'm2', 'm4']);
    expect(result.current.runsData?.paramKeyList).toEqual(['p1', 'p4']);

    // Check that the next page is not available
    expect(result.current.moreRunsAvailable).toEqual(false);
  });

  test('should set an error if there is any', async () => {
    // Mock the response to be a failure
    jest.mocked(searchRunsApi).mockReturnValue({
      type: 'mocked-action',
      payload: Promise.reject(new ErrorWrapper({ message: 'request failure' })),
      meta: { id: 0 },
    });
    jest.spyOn(Utils, 'logErrorAndNotifyUser').mockImplementation(() => {});
    const { result } = await renderTestHook();
    expect(result.current.requestError?.getMessageField()).toEqual('request failure');
    expect(Utils.logErrorAndNotifyUser).toHaveBeenCalled();
  });
});
