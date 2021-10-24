import React from 'react';
import { shallow } from 'enzyme';
import { MemoryRouter as Router } from 'react-router-dom';

import { ErrorCodes } from '../../common/constants';
import { ExperimentPage, isNewRun, lifecycleFilterToRunViewType } from './ExperimentPage';
import ExperimentView from './ExperimentView';
import { PermissionDeniedView } from './PermissionDeniedView';
import { ViewType } from '../sdk/MlflowEnums';
import { ErrorWrapper, getUUID } from '../../common/utils/ActionUtils';
import { MAX_RUNS_IN_SEARCH_MODEL_VERSIONS_FILTER } from '../../model-registry/constants';
import {
  ATTRIBUTE_COLUMN_SORT_KEY,
  DETECT_NEW_RUNS_INTERVAL,
  MAX_DETECT_NEW_RUNS_RESULTS,
  PAGINATION_DEFAULT_STATE,
  DEFAULT_ORDER_BY_KEY,
  DEFAULT_ORDER_BY_ASC,
  DEFAULT_START_TIME,
} from '../constants';

const EXPERIMENT_ID = '17';

jest.useFakeTimers();

let searchRunsApi;
let getExperimentApi;
let loadMoreRunsApi;
let searchModelVersionsApi;
let searchForNewRuns;
let history;
let location;

beforeEach(() => {
  searchRunsApi = jest.fn(() => Promise.resolve());
  getExperimentApi = jest.fn(() => Promise.resolve());
  searchModelVersionsApi = jest.fn(() => Promise.resolve());
  loadMoreRunsApi = jest.fn(() => Promise.resolve());
  searchForNewRuns = jest.fn(() => Promise.resolve());
  location = {};
  history = {};
});

const getExperimentPageMock = (additionalProps) => {
  return shallow(
    <ExperimentPage
      experimentId={EXPERIMENT_ID}
      searchRunsApi={searchRunsApi}
      getExperimentApi={getExperimentApi}
      searchModelVersionsApi={searchModelVersionsApi}
      loadMoreRunsApi={loadMoreRunsApi}
      searchForNewRuns={searchForNewRuns}
      history={history}
      location={location}
      {...additionalProps}
    />,
  );
};

test('State and search params are correct for blank search', () => {
  const wrapper = getExperimentPageMock();
  wrapper.instance().onSearch('', 'Active', null, true, null);

  expect(wrapper.state().persistedState.searchInput).toEqual('');
  expect(wrapper.state().lifecycleFilter).toEqual('Active');
  expect(wrapper.state().persistedState.orderByKey).toEqual(null);
  expect(wrapper.state().persistedState.orderByAsc).toEqual(true);
  expect(wrapper.state().modelVersionFilter).toEqual(null);
  expect(wrapper.state().persistedState.startTime).toEqual(undefined);

  const searchRunsCallParams = searchRunsApi.mock.calls[1][0];

  expect(searchRunsCallParams.experimentIds).toEqual([EXPERIMENT_ID]);
  expect(searchRunsCallParams.filter).toEqual('');
  expect(searchRunsCallParams.runViewType).toEqual(ViewType.ACTIVE_ONLY);
  expect(searchRunsCallParams.orderBy).toEqual([]);
});

test('State and search params are correct for complete search', () => {
  const wrapper = getExperimentPageMock();
  wrapper.instance().onSearch('metrics.metric0 > 3', 'Deleted', null, true, null, 'ALL');

  expect(wrapper.state().persistedState.searchInput).toEqual('metrics.metric0 > 3');
  expect(wrapper.state().lifecycleFilter).toEqual('Deleted');
  expect(wrapper.state().persistedState.orderByKey).toEqual(null);
  expect(wrapper.state().persistedState.orderByAsc).toEqual(true);
  expect(wrapper.state().modelVersionFilter).toEqual(null);
  expect(wrapper.state().persistedState.startTime).toEqual('ALL');

  const searchRunsCallParams = searchRunsApi.mock.calls[1][0];
  expect(searchRunsCallParams.filter).toEqual('metrics.metric0 > 3');
  expect(searchRunsCallParams.runViewType).toEqual(ViewType.DELETED_ONLY);
});

test('State and search params are correct for search with order_by', () => {
  const wrapper = getExperimentPageMock();
  wrapper.instance().onSearch('', 'Active', 'my_key', false, null);

  expect(wrapper.state().persistedState.searchInput).toEqual('');
  expect(wrapper.state().lifecycleFilter).toEqual('Active');
  expect(wrapper.state().persistedState.orderByKey).toEqual('my_key');
  expect(wrapper.state().persistedState.orderByAsc).toEqual(false);
  expect(wrapper.state().modelVersionFilter).toEqual(null);
  expect(wrapper.state().persistedState.startTime).toEqual(undefined);

  const searchRunsCallParams = searchRunsApi.mock.calls[1][0];
  expect(searchRunsCallParams.filter).toEqual('');
  expect(searchRunsCallParams.orderBy).toEqual(['my_key DESC']);
});

test('Loading state without any URL params', () => {
  const wrapper = getExperimentPageMock();
  const { state } = wrapper.instance();
  expect(state.persistedState.searchInput).toEqual('');
  expect(state.persistedState.orderByKey).toBe(DEFAULT_ORDER_BY_KEY);
  expect(state.persistedState.orderByAsc).toEqual(DEFAULT_ORDER_BY_ASC);
  expect(state.persistedState.startTime).toEqual(DEFAULT_START_TIME);
});

test('Loading state with all URL params', () => {
  location.search = 'search=c&orderByKey=d&orderByAsc=false&startTime=LAST_HOUR';
  const wrapper = getExperimentPageMock();
  const { state } = wrapper.instance();
  expect(state.persistedState.searchInput).toEqual('c');
  expect(state.persistedState.orderByKey).toEqual('d');
  expect(state.persistedState.orderByAsc).toEqual(false);
  expect(state.persistedState.startTime).toEqual('LAST_HOUR');
});

test('should render permission denied view when getExperiment yields permission error', () => {
  const experimentPageInstance = getExperimentPageMock().instance();
  experimentPageInstance.setState({
    getExperimentRequestId: getUUID(),
    searchRunsRequestId: getUUID(),
  });
  const errorMessage = 'Access Denied';
  const responseErrorWrapper = new ErrorWrapper({
    responseText: `{"error_code": "${ErrorCodes.PERMISSION_DENIED}", "message": "${errorMessage}"}`,
  });
  const searchRunsErrorRequest = {
    id: experimentPageInstance.state.searchRunsRequestId,
    active: false,
    error: responseErrorWrapper,
  };
  const getExperimentErrorRequest = {
    id: experimentPageInstance.state.getExperimentRequestId,
    active: false,
    error: responseErrorWrapper,
  };
  const experimentViewInstance = shallow(
    experimentPageInstance.renderExperimentView(false, true, [
      searchRunsErrorRequest,
      getExperimentErrorRequest,
    ]),
  ).instance();
  expect(experimentViewInstance).toBeInstanceOf(PermissionDeniedView);
  expect(experimentViewInstance.props.errorMessage).toEqual(errorMessage);
});

test('should render experiment view when search error occurs', () => {
  const experimentPageInstance = getExperimentPageMock().instance();
  experimentPageInstance.setState({
    getExperimentRequestId: getUUID(),
    searchRunsRequestId: getUUID(),
  });
  const responseErrorWrapper = new ErrorWrapper({
    responseText: `{"error_code": "${ErrorCodes.INVALID_PARAMETER_VALUE}", "message": "Invalid"}`,
  });
  const searchRunsErrorRequest = {
    id: experimentPageInstance.state.searchRunsRequestId,
    active: false,
    error: responseErrorWrapper,
  };
  const getExperimentErrorRequest = {
    id: experimentPageInstance.state.getExperimentRequestId,
    active: false,
  };
  const renderedView = shallow(
    <Router>
      {experimentPageInstance.renderExperimentView(false, true, [
        searchRunsErrorRequest,
        getExperimentErrorRequest,
      ])}
    </Router>,
  );
  expect(renderedView.find(ExperimentView)).toHaveLength(1);
});

test('should update next page token initially', () => {
  const promise = Promise.resolve({ value: { next_page_token: 'token_1' } });
  searchRunsApi = jest.fn(() => promise);
  const wrapper = getExperimentPageMock();
  const instance = wrapper.instance();
  return promise.then(() => expect(instance.state.nextPageToken).toBe('token_1'));
});

test('should update next page token after load-more', () => {
  const promise = Promise.resolve({ value: { next_page_token: 'token_1' } });
  loadMoreRunsApi = jest.fn(() => promise);
  const wrapper = getExperimentPageMock();
  const instance = wrapper.instance();
  instance.handleLoadMoreRuns();
  return promise.then(() => expect(instance.state.nextPageToken).toBe('token_1'));
});

test('should update next page token to null when load-more response has no token', () => {
  const promise1 = Promise.resolve({ value: { next_page_token: 'token_1' } });
  const promise2 = Promise.resolve({ value: {} });
  searchRunsApi = jest.fn(() => promise1);
  loadMoreRunsApi = jest.fn(() => promise2);
  const wrapper = getExperimentPageMock();
  const instance = wrapper.instance();
  instance.handleLoadMoreRuns();
  return Promise.all([promise1, promise2]).then(() =>
    expect(instance.state.nextPageToken).toBe(null),
  );
});

test('should set state to default values on promise rejection when loading more', () => {
  loadMoreRunsApi = jest.fn(() => Promise.reject());
  const wrapper = getExperimentPageMock();
  const instance = wrapper.instance();
  return Promise.resolve(instance.handleLoadMoreRuns()).then(() => {
    expect(instance.state.nextPageToken).toBe(PAGINATION_DEFAULT_STATE.nextPageToken);
    expect(instance.state.numRunsFromLatestSearch).toBe(
      PAGINATION_DEFAULT_STATE.numRunsFromLatestSearch,
    );
    expect(instance.state.loadingMore).toBe(PAGINATION_DEFAULT_STATE.loadingMore);
  });
});

test('should set state to default values on promise rejection onSearch', () => {
  searchRunsApi = jest.fn(() => Promise.reject());
  const wrapper = getExperimentPageMock();
  const instance = wrapper.instance();
  return Promise.resolve(instance.onSearch()).then(() => {
    expect(instance.state.nextPageToken).toBe(PAGINATION_DEFAULT_STATE.nextPageToken);
    expect(instance.state.numRunsFromLatestSearch).toBe(
      PAGINATION_DEFAULT_STATE.numRunsFromLatestSearch,
    );
    expect(instance.state.loadingMore).toBe(PAGINATION_DEFAULT_STATE.loadingMore);
  });
});

test('should nest children when filtering or sorting', () => {
  const wrapper = getExperimentPageMock();
  const instance = wrapper.instance();

  instance.setState(
    {
      persistedState: {
        orderByKey: null,
        searchInput: null,
      },
    },
    () => expect(instance.shouldNestChildrenAndFetchParents()).toBe(true),
  );
  instance.setState(
    {
      persistedState: {
        orderByKey: 'name',
        searchInput: null,
      },
    },
    () => expect(instance.shouldNestChildrenAndFetchParents()).toBe(false),
  );
  instance.setState(
    {
      persistedState: {
        orderByKey: null,
        searchInput: 'metrics.a > 1',
      },
    },
    () => expect(instance.shouldNestChildrenAndFetchParents()).toBe(false),
  );
  instance.setState(
    {
      persistedState: {
        orderByKey: 'name',
        searchInput: 'metrics.a > 1',
      },
    },
    () => expect(instance.shouldNestChildrenAndFetchParents()).toBe(false),
  );
  instance.setState(
    {
      persistedState: {
        orderByKey: ATTRIBUTE_COLUMN_SORT_KEY.DATE,
        searchInput: 'metrics.a > 1',
      },
    },
    () => expect(instance.shouldNestChildrenAndFetchParents()).toBe(true),
  );
  instance.setState(
    {
      persistedState: {
        orderByKey: ATTRIBUTE_COLUMN_SORT_KEY.DATE,
        searchInput: null,
      },
    },
    () => expect(instance.shouldNestChildrenAndFetchParents()).toBe(true),
  );
});

test('should return correct orderBy expression', () => {
  const wrapper = getExperimentPageMock();
  const instance = wrapper.instance();

  instance.setState(
    {
      persistedState: {
        orderByKey: 'key',
        orderByAsc: true,
      },
    },
    () => expect(instance.getOrderByExpr()).toEqual(['key ASC']),
  );
  instance.setState(
    {
      persistedState: {
        orderByKey: 'key',
        orderByAsc: false,
      },
    },
    () => expect(instance.getOrderByExpr()).toEqual(['key DESC']),
  );
  instance.setState(
    {
      persistedState: {
        orderByKey: '',
        orderByAsc: true,
      },
    },
    () => expect(instance.getOrderByExpr()).toEqual([]),
  );
  instance.setState(
    {
      persistedState: {
        orderByKey: null,
        orderByAsc: null,
      },
    },
    () => expect(instance.getOrderByExpr()).toEqual([]),
  );
});

test('handleGettingRuns chain functions should not change response', () => {
  const wrapper = getExperimentPageMock();
  const instance = wrapper.instance();

  const response = {
    value: {
      runs: [
        {
          info: {},
          data: {},
        },
      ],
    },
  };

  expect(instance.updateNextPageToken(response)).toEqual(response);
  expect(instance.updateNumRunsFromLatestSearch(response)).toEqual(response);
  expect(instance.fetchModelVersionsForRuns(response)).toEqual(response);
});

describe('updateNextPageToken', () => {
  it('should set loadingMore to false and update nextPageToken', () => {
    const wrapper = getExperimentPageMock();
    const instance = wrapper.instance();

    instance.updateNextPageToken({ value: { next_page_token: 'token' } });
    expect(instance.state.nextPageToken).toBe('token');
    expect(instance.state.loadingMore).toBe(false);
  });

  it('should set nextPageToken to null when not given one', () => {
    const wrapper = getExperimentPageMock();
    const instance = wrapper.instance();

    instance.updateNextPageToken({});
    expect(instance.state.nextPageToken).toBe(null);
    expect(instance.state.loadingMore).toBe(false);
  });
});

describe('updateNumRunsFromLatestSearch', () => {
  test('should update numRunsFromLatestSearch correctly', () => {
    const wrapper = getExperimentPageMock();
    const instance = wrapper.instance();

    const runs = Array(100).fill([{ info: {}, data: {} }]);
    instance.updateNumRunsFromLatestSearch({ value: { runs } });

    expect(instance.state.numRunsFromLatestSearch).toBe(100);
  });

  test('should not update if no runs', () => {
    const wrapper = getExperimentPageMock();
    const instance = wrapper.instance();
    instance.setState({ numRunsFromLatestSearch: 1 });
    instance.updateNumRunsFromLatestSearch({});
    expect(instance.state.numRunsFromLatestSearch).toBe(1);
    instance.updateNumRunsFromLatestSearch({ value: {} });
    expect(instance.state.numRunsFromLatestSearch).toBe(1);
  });
});

describe('fetchModelVersionsForRuns', () => {
  it('when given valid response, should call searchModelVersionsApi with correct arguments', () => {
    const wrapper = getExperimentPageMock();
    const instance = wrapper.instance();

    instance.fetchModelVersionsForRuns({
      value: {
        runs: [{ info: { run_id: '1' } }, { info: { run_id: '2' } }, { info: { run_id: '3' } }],
      },
    });

    expect(searchModelVersionsApi).toHaveBeenCalledWith(
      { run_id: ['1', '2', '3'] },
      instance.searchModelVersionsRequestId,
    );
  });

  it('should not call searchModelVersionsApi if invalid or no runs', () => {
    const wrapper = getExperimentPageMock();
    const instance = wrapper.instance();

    instance.fetchModelVersionsForRuns();
    instance.fetchModelVersionsForRuns({});
    instance.fetchModelVersionsForRuns({ value: {} });
    instance.fetchModelVersionsForRuns({ value: { runs: [] } });

    expect(searchModelVersionsApi).not.toHaveBeenCalled();
  });

  it('should chunk runs to searchModelVersions', () => {
    const wrapper = getExperimentPageMock();
    const instance = wrapper.instance();
    const runs = [...Array(MAX_RUNS_IN_SEARCH_MODEL_VERSIONS_FILTER + 1).keys()].map((run_id) => ({
      info: { run_id },
      data: {},
    }));

    instance.fetchModelVersionsForRuns({ value: { runs } });

    expect(searchModelVersionsApi).toHaveBeenCalledTimes(2);
  });
});

describe('handleGettingRuns', () => {
  it('should call updateNextPageToken, updateNumRunsFromLatestSearch, fetchModelVersionsForRuns', () => {
    const wrapper = getExperimentPageMock();
    const instance = wrapper.instance();
    instance.updateNextPageToken = jest.fn();
    instance.updateNumRunsFromLatestSearch = jest.fn();
    instance.fetchModelVersionsForRuns = jest.fn();

    return Promise.resolve(
      instance.handleGettingRuns(() => Promise.resolve(), instance.searchRunsApi),
    ).then(() => {
      expect(instance.updateNextPageToken).toHaveBeenCalled();
      expect(instance.updateNumRunsFromLatestSearch).toHaveBeenCalled();
      expect(instance.fetchModelVersionsForRuns).toHaveBeenCalled();
    });
  });
});

test('lifecycleFilterToRunViewType', () => {
  expect(lifecycleFilterToRunViewType('Active')).toBe('ACTIVE_ONLY');
  expect(lifecycleFilterToRunViewType('Deleted')).toBe('DELETED_ONLY');
});

describe('detectNewRuns', () => {
  describe('refresh behaviour', () => {
    test('Should refresh once after DETECT_NEW_RUNS_INTERVAL', () => {
      getExperimentPageMock();
      jest.advanceTimersByTime(DETECT_NEW_RUNS_INTERVAL - 1);
      expect(searchForNewRuns).toHaveBeenCalledTimes(0);
      jest.advanceTimersByTime(1);
      expect(searchForNewRuns).toHaveBeenCalledTimes(1);
    });

    test('Should refresh every DETECT_NEW_RUNS_INTERVAL', () => {
      getExperimentPageMock();
      jest.advanceTimersByTime(DETECT_NEW_RUNS_INTERVAL - 1);
      expect(searchForNewRuns).toHaveBeenCalledTimes(0);
      jest.advanceTimersByTime(1);
      expect(searchForNewRuns).toHaveBeenCalledTimes(1);
      jest.advanceTimersByTime(DETECT_NEW_RUNS_INTERVAL);
      expect(searchForNewRuns).toHaveBeenCalledTimes(2);
    });

    test('Should not keep refreshing after unmount', () => {
      const mock = getExperimentPageMock();

      mock.unmount();

      jest.advanceTimersByTime(DETECT_NEW_RUNS_INTERVAL);
      expect(searchForNewRuns).toHaveBeenCalledTimes(0);
      jest.advanceTimersByTime(DETECT_NEW_RUNS_INTERVAL);
      expect(searchForNewRuns).toHaveBeenCalledTimes(0);
      jest.advanceTimersByTime(DETECT_NEW_RUNS_INTERVAL * 100);
      expect(searchForNewRuns).toHaveBeenCalledTimes(0);
    });

    describe('Interval clearing behaviour', () => {
      const maxNewRuns = [];
      for (let i = 0; i < MAX_DETECT_NEW_RUNS_RESULTS; i++) {
        maxNewRuns.push({ info: { start_time: Date.now() + 10000 } });
      }

      test('Should stop polling if there are already max new runs', async () => {
        const mockSearchForNewRuns = jest.fn(() => Promise.resolve({ runs: maxNewRuns }));

        const instance = getExperimentPageMock({
          searchForNewRuns: mockSearchForNewRuns,
        }).instance();

        expect(mockSearchForNewRuns).toHaveBeenCalledTimes(0);
        await instance.detectNewRuns();
        expect(mockSearchForNewRuns).toHaveBeenCalledTimes(1);
        expect(instance.detectNewRunsTimer).toEqual(null);
        jest.advanceTimersByTime(DETECT_NEW_RUNS_INTERVAL * 100);
        expect(mockSearchForNewRuns).toHaveBeenCalledTimes(1);
      });

      test('Should resume polling if a new search is triggered', async () => {
        const mockSearchForNewRuns = jest.fn(() => Promise.resolve({ runs: maxNewRuns }));

        const instance = getExperimentPageMock({
          searchForNewRuns: mockSearchForNewRuns,
        }).instance();

        await instance.detectNewRuns();
        expect(mockSearchForNewRuns).toHaveBeenCalledTimes(1);
        expect(instance.detectNewRunsTimer).toEqual(null);
        jest.advanceTimersByTime(DETECT_NEW_RUNS_INTERVAL * 100);
        expect(mockSearchForNewRuns).toHaveBeenCalledTimes(1);

        await instance.onSearch();

        jest.advanceTimersByTime(DETECT_NEW_RUNS_INTERVAL);
        expect(mockSearchForNewRuns).toHaveBeenCalledTimes(2);
        jest.advanceTimersByTime(DETECT_NEW_RUNS_INTERVAL);
        expect(mockSearchForNewRuns).toHaveBeenCalledTimes(3);
      });
    });
  });

  describe('numberOfNewRuns state', () => {
    test('numberOfNewRuns should be 0 be default', () => {
      const instance = getExperimentPageMock().instance();
      expect(instance.state.numberOfNewRuns).toEqual(0);
    });

    test('numberOfNewRuns should be 0 if no new runs', async () => {
      const instance = getExperimentPageMock({
        searchForNewRuns: () => Promise.resolve({ runs: [] }),
      }).instance();

      await instance.detectNewRuns();
      expect(instance.state.numberOfNewRuns).toEqual(0);
    });

    test('Should update numberOfNewRuns correctly', async () => {
      const mockSearchForNewRuns = jest.fn(() =>
        Promise.resolve({
          runs: [
            {
              info: {
                start_time: Date.now() + 10000,
              },
            },
            {
              info: {
                end_time: Date.now() + 10000,
              },
            },
            {
              info: {
                end_time: 0,
              },
            },
          ],
        }),
      );

      const instance = getExperimentPageMock({
        searchForNewRuns: mockSearchForNewRuns,
      }).instance();

      await instance.detectNewRuns();
      expect(instance.state.numberOfNewRuns).toEqual(2);
    });

    test('Should not explode if no runs', async () => {
      const instance = getExperimentPageMock().instance();

      await instance.detectNewRuns();
      expect(instance.state.numberOfNewRuns).toEqual(0);
    });
  });
});

describe('isNewRun', () => {
  test('should return false if run undefined', () => {
    expect(isNewRun(2, undefined)).toEqual(false);
  });

  test('should return false if run info undefined', () => {
    expect(isNewRun(2, {})).toEqual(false);
  });

  test('should return false if start time and end time undefined', () => {
    expect(
      isNewRun(2, {
        info: {},
      }),
    ).toEqual(false);
  });

  test('should return false if start time and end time are < lastRunsRefreshTime', () => {
    expect(
      isNewRun(2, {
        info: {
          start_time: 1,
          end_time: 1,
        },
      }),
    ).toEqual(false);
  });

  test('should return false if start time < lastRunsRefreshTime and end time is 0', () => {
    expect(
      isNewRun(2, {
        info: {
          start_time: 1,
          end_time: 0,
        },
      }),
    ).toEqual(false);
  });

  test('should return true if start time >= lastRunsRefreshTime', () => {
    expect(
      isNewRun(1, {
        info: {
          start_time: 1,
          end_time: 0,
        },
      }),
    ).toEqual(true);
  });

  test('should return true if end time not 0 and <= lastRunsRefreshTime', () => {
    expect(
      isNewRun(2, {
        info: {
          start_time: 1,
          end_time: 3,
        },
      }),
    ).toEqual(true);
  });
});

describe('startTime select filters out the experiment runs correctly', () => {
  test('should get startTime expr for the filter query generated correctly', () => {
    const wrapper = getExperimentPageMock();
    const instance = wrapper.instance();

    instance.setState(
      {
        persistedState: {
          startTime: '',
        },
      },
      () => expect(instance.getStartTimeExpr()).toBe(null),
    );

    instance.setState(
      {
        persistedState: {
          startTime: undefined,
        },
      },
      () => expect(instance.getStartTimeExpr()).toBe(null),
    );

    instance.setState(
      {
        persistedState: {
          startTime: 'ALL',
        },
      },
      () => expect(instance.getStartTimeExpr()).toBe(null),
    );

    instance.setState(
      {
        persistedState: {
          startTime: 'LAST_24_HOURS',
        },
      },
      () => expect(instance.getStartTimeExpr()).toMatch('attributes.start_time'),
    );
  });

  test('handleGettingRuns correctly generates the filter string', () => {
    const wrapper = getExperimentPageMock();
    const instance = wrapper.instance();
    const getRunsAction = jest.fn(() => Promise.resolve());
    const requestId = '123';

    instance.setState(
      {
        persistedState: {
          startTime: '',
          searchInput: 'metrics.met > 0',
        },
      },
      () => {
        instance.handleGettingRuns(getRunsAction, requestId);
        expect(getRunsAction).toHaveBeenCalledWith(
          expect.objectContaining({
            filter: 'metrics.met > 0',
          }),
        );
      },
    );

    instance.setState(
      {
        persistedState: {
          startTime: 'ALL',
          searchInput: 'metrics.met > 0',
        },
      },
      () => {
        instance.handleGettingRuns(getRunsAction, requestId);
        expect(getRunsAction).toHaveBeenCalledWith(
          expect.objectContaining({
            filter: 'metrics.met > 0',
          }),
        );
      },
    );

    instance.setState(
      {
        persistedState: {
          startTime: 'LAST_HOUR_FAKE',
          searchInput: 'metrics.met > 0',
        },
      },
      () => {
        instance.handleGettingRuns(getRunsAction, requestId);
        expect(getRunsAction).toHaveBeenCalledWith(
          expect.objectContaining({
            filter: 'metrics.met > 0',
          }),
        );
      },
    );

    instance.setState(
      {
        persistedState: {
          startTime: 'LAST_HOUR',
          searchInput: 'metrics.met > 0',
        },
      },
      () => {
        instance.handleGettingRuns(getRunsAction, requestId);
        expect(getRunsAction).toHaveBeenCalledWith(
          expect.objectContaining({
            filter: expect.stringMatching('metrics.met > 0 and attributes.start_time'),
          }),
        );
      },
    );
  });
});
