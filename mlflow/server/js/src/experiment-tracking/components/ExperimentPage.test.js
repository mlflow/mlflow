import React from 'react';
import { shallow } from 'enzyme';
import { MemoryRouter as Router } from 'react-router-dom';

import { ErrorCodes } from '../../common/constants';
import { ExperimentPage, isNewRun, lifecycleFilterToRunViewType } from './ExperimentPage';
import { ExperimentPagePersistedState } from '../sdk/MlflowLocalStorageMessages';
import Utils from '../../common/utils/Utils';
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
  DEFAULT_LIFECYCLE_FILTER,
  DEFAULT_MODEL_VERSION_FILTER,
  MODEL_VERSION_FILTER,
  DEFAULT_CATEGORIZED_UNCHECKED_KEYS,
  DEFAULT_SHOW_MULTI_COLUMNS,
  DEFAULT_DIFF_SWITCH_SELECTED,
  COLUMN_TYPES,
  LIFECYCLE_FILTER,
} from '../constants';

const EXPERIMENT_ID = '17';
const BASE_PATH = '/experiments/17/s';

jest.useFakeTimers();

let searchRunsApi;
let getExperimentApi;
let batchGetExperimentsApi;
let loadMoreRunsApi;
let searchModelVersionsApi;
let searchForNewRuns;
let history;
let location;

beforeEach(() => {
  localStorage.clear();
  searchRunsApi = jest.fn(() => Promise.resolve());
  getExperimentApi = jest.fn(() => Promise.resolve());
  batchGetExperimentsApi = jest.fn(() => Promise.resolve());
  searchModelVersionsApi = jest.fn(() => Promise.resolve());
  loadMoreRunsApi = jest.fn(() => Promise.resolve());
  searchForNewRuns = jest.fn(() => Promise.resolve());
  location = {
    pathname: '/',
  };
  history = {
    push: jest.fn(),
    location: {
      pathname: BASE_PATH,
      search: '',
    },
  };
});

const getExperimentPageMock = (additionalProps) => {
  return shallow(
    <ExperimentPage
      experimentId={EXPERIMENT_ID}
      searchRunsApi={searchRunsApi}
      getExperimentApi={getExperimentApi}
      batchGetExperimentsApi={batchGetExperimentsApi}
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
  const wrapper = getExperimentPageMock({
    location: {
      search: '?searchInput=test',
    },
  });
  wrapper.instance().onSearch({ searchInput: '' });

  expect(wrapper.state().persistedState.searchInput).toEqual('');
  expect(wrapper.state().persistedState.orderByKey).toEqual(DEFAULT_ORDER_BY_KEY);
  expect(wrapper.state().persistedState.orderByAsc).toEqual(DEFAULT_ORDER_BY_ASC);
  expect(wrapper.state().persistedState.lifecycleFilter).toEqual(DEFAULT_LIFECYCLE_FILTER);
  expect(wrapper.state().persistedState.modelVersionFilter).toEqual(DEFAULT_MODEL_VERSION_FILTER);
  expect(wrapper.state().persistedState.startTime).toEqual(DEFAULT_START_TIME);

  const searchRunsCallParams = searchRunsApi.mock.calls[1][0];

  expect(searchRunsCallParams.experimentIds).toEqual([EXPERIMENT_ID]);
  expect(searchRunsCallParams.filter).toEqual('');
  expect(searchRunsCallParams.runViewType).toEqual(ViewType.ACTIVE_ONLY);
  expect(searchRunsCallParams.orderBy).toEqual(['attributes.start_time DESC']);
});

test('State and search params are correct for complete search', () => {
  const wrapper = getExperimentPageMock();
  wrapper.instance().onSearch({
    searchInput: 'metrics.metric0 > 3',
    orderByKey: 'test-key',
    orderByAsc: true,
    lifecycleFilter: 'Deleted',
    modelVersionFilter: MODEL_VERSION_FILTER.WTIHOUT_MODEL_VERSIONS,
    startTime: '1 Hour',
  });

  expect(wrapper.state().persistedState.searchInput).toEqual('metrics.metric0 > 3');
  expect(wrapper.state().persistedState.orderByKey).toEqual('test-key');
  expect(wrapper.state().persistedState.orderByAsc).toEqual(true);
  expect(wrapper.state().persistedState.lifecycleFilter).toEqual('Deleted');
  expect(wrapper.state().persistedState.modelVersionFilter).toEqual(
    MODEL_VERSION_FILTER.WTIHOUT_MODEL_VERSIONS,
  );
  expect(wrapper.state().persistedState.startTime).toEqual('1 Hour');

  const searchRunsCallParams = searchRunsApi.mock.calls[1][0];
  expect(searchRunsCallParams.filter).toEqual('metrics.metric0 > 3');
  expect(searchRunsCallParams.runViewType).toEqual(ViewType.DELETED_ONLY);
  expect(searchRunsCallParams.orderBy).toEqual(['test-key ASC']);
});

test('Loading state without any URL params and no snapshot', () => {
  const wrapper = getExperimentPageMock();
  const { state } = wrapper.instance();
  expect(state.persistedState.searchInput).toEqual('');
  expect(state.persistedState.lifecycleFilter).toEqual(DEFAULT_LIFECYCLE_FILTER);
  expect(state.persistedState.modelVersionFilter).toEqual(DEFAULT_MODEL_VERSION_FILTER);
  expect(state.persistedState.orderByKey).toBe(DEFAULT_ORDER_BY_KEY);
  expect(state.persistedState.orderByAsc).toEqual(DEFAULT_ORDER_BY_ASC);
  expect(state.persistedState.startTime).toEqual(DEFAULT_START_TIME);
  expect(state.persistedState.showMultiColumns).toEqual(DEFAULT_SHOW_MULTI_COLUMNS);
  expect(state.persistedState.diffSwitchSelected).toEqual(DEFAULT_DIFF_SWITCH_SELECTED);
  expect(state.persistedState.categorizedUncheckedKeys).toEqual(DEFAULT_CATEGORIZED_UNCHECKED_KEYS);
  expect(state.persistedState.preSwitchCategorizedUncheckedKeys).toEqual(
    DEFAULT_CATEGORIZED_UNCHECKED_KEYS,
  );
  expect(state.persistedState.postSwitchCategorizedUncheckedKeys).toEqual(
    DEFAULT_CATEGORIZED_UNCHECKED_KEYS,
  );
});

test('Loading state with all URL params and no snapshot', () => {
  location.search =
    'searchInput=c&orderByKey=d&orderByAsc=false&startTime=LAST_HOUR' +
    '&lifecycleFilter=lifecycle&modelVersionFilter=With%20Model%20Versions' +
    '&showMultiColumns=false' +
    '&diffSwitchSelected=true' +
    '&categorizedUncheckedKeys%5Battributes%5D%5B0%5D=a1' +
    '&categorizedUncheckedKeys%5Bparams%5D%5B0%5D=p1' +
    '&categorizedUncheckedKeys%5Bmetrics%5D%5B0%5D=m1' +
    '&categorizedUncheckedKeys%5Btags%5D%5B0%5D=t1' +
    '&preSwitchCategorizedUncheckedKeys%5Battributes%5D%5B0%5D=a2' +
    '&preSwitchCategorizedUncheckedKeys%5Bparams%5D%5B0%5D=p2' +
    '&preSwitchCategorizedUncheckedKeys%5Bmetrics%5D%5B0%5D=m2' +
    '&preSwitchCategorizedUncheckedKeys%5Btags%5D%5B0%5D=t2' +
    '&postSwitchCategorizedUncheckedKeys%5Battributes%5D%5B0%5D=a3' +
    '&postSwitchCategorizedUncheckedKeys%5Bparams%5D%5B0%5D=p3' +
    '&postSwitchCategorizedUncheckedKeys%5Bmetrics%5D%5B0%5D=m3' +
    '&postSwitchCategorizedUncheckedKeys%5Btags%5D%5B0%5D=t3';

  const wrapper = getExperimentPageMock();
  const { state } = wrapper.instance();
  expect(state.persistedState.searchInput).toEqual('c');
  expect(state.persistedState.lifecycleFilter).toEqual('lifecycle');
  expect(state.persistedState.modelVersionFilter).toEqual('With Model Versions');
  expect(state.persistedState.orderByKey).toEqual('d');
  expect(state.persistedState.orderByAsc).toEqual(false);
  expect(state.persistedState.startTime).toEqual('LAST_HOUR');
  expect(state.persistedState.showMultiColumns).toEqual(false);
  expect(state.persistedState.diffSwitchSelected).toEqual(true);
  expect(state.persistedState.categorizedUncheckedKeys).toEqual({
    [COLUMN_TYPES.ATTRIBUTES]: ['a1'],
    [COLUMN_TYPES.PARAMS]: ['p1'],
    [COLUMN_TYPES.METRICS]: ['m1'],
    [COLUMN_TYPES.TAGS]: ['t1'],
  });
  expect(state.persistedState.preSwitchCategorizedUncheckedKeys).toEqual({
    [COLUMN_TYPES.ATTRIBUTES]: ['a2'],
    [COLUMN_TYPES.PARAMS]: ['p2'],
    [COLUMN_TYPES.METRICS]: ['m2'],
    [COLUMN_TYPES.TAGS]: ['t2'],
  });
  expect(state.persistedState.postSwitchCategorizedUncheckedKeys).toEqual({
    [COLUMN_TYPES.ATTRIBUTES]: ['a3'],
    [COLUMN_TYPES.PARAMS]: ['p3'],
    [COLUMN_TYPES.METRICS]: ['m3'],
    [COLUMN_TYPES.TAGS]: ['t3'],
  });
});

test('onClear clears all parameters', () => {
  const wrapper = getExperimentPageMock();
  const instance = wrapper.instance();
  const updateUrlWithViewStateSpy = jest.fn();
  instance.updateUrlWithViewState = updateUrlWithViewStateSpy;
  instance.setState({
    persistedState: new ExperimentPagePersistedState({
      searchInput: 'testing',
      orderByKey: 'test-key',
      orderByAsc: false,
      startTime: 'HOUR',
      lifecycleFilter: LIFECYCLE_FILTER.DELETED,
      modelVersionFilter: MODEL_VERSION_FILTER.WITH_MODEL_VERSIONS,
      showMultiColumns: false,
      categorizedUncheckedKeys: {},
      diffSwitchSelected: true,
      preSwitchCategorizedUncheckedKeys: {},
      postSwitchCategorizedUncheckedKeys: {},
    }).toJSON(),
  });

  instance.onClear();
  const { state } = instance;
  expect(updateUrlWithViewStateSpy).toHaveBeenCalledTimes(1);
  expect(state.persistedState.searchInput).toEqual('');
  expect(state.persistedState.lifecycleFilter).toEqual(DEFAULT_LIFECYCLE_FILTER);
  expect(state.persistedState.modelVersionFilter).toEqual(DEFAULT_MODEL_VERSION_FILTER);
  expect(state.persistedState.orderByKey).toBe(DEFAULT_ORDER_BY_KEY);
  expect(state.persistedState.orderByAsc).toEqual(DEFAULT_ORDER_BY_ASC);
  expect(state.persistedState.startTime).toEqual(DEFAULT_START_TIME);
  expect(state.persistedState.showMultiColumns).toEqual(false);
  expect(state.persistedState.diffSwitchSelected).toEqual(DEFAULT_DIFF_SWITCH_SELECTED);
  expect(state.persistedState.categorizedUncheckedKeys).toEqual(DEFAULT_CATEGORIZED_UNCHECKED_KEYS);
  expect(state.persistedState.preSwitchCategorizedUncheckedKeys).toEqual(
    DEFAULT_CATEGORIZED_UNCHECKED_KEYS,
  );
  expect(state.persistedState.postSwitchCategorizedUncheckedKeys).toEqual(
    DEFAULT_CATEGORIZED_UNCHECKED_KEYS,
  );
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
  return Promise.resolve(instance.onSearch({})).then(() => {
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

        await instance.onSearch({});

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
        persistedState: new ExperimentPagePersistedState({
          startTime: '',
        }).toJSON(),
      },
      () => expect(instance.getStartTimeExpr()).toBe(null),
    );

    instance.setState(
      {
        persistedState: new ExperimentPagePersistedState({
          startTime: undefined,
        }).toJSON(),
      },
      () => expect(instance.getStartTimeExpr()).toBe(null),
    );

    instance.setState(
      {
        persistedState: new ExperimentPagePersistedState({
          startTime: 'ALL',
        }).toJSON(),
      },
      () => expect(instance.getStartTimeExpr()).toBe(null),
    );

    instance.setState(
      {
        persistedState: new ExperimentPagePersistedState({
          startTime: 'LAST_24_HOURS',
        }).toJSON(),
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

function expectSearchState(historyEntry, searchQueryParams) {
  const expectedPrefix = BASE_PATH + '?';
  expect(historyEntry.startsWith(expectedPrefix)).toBe(true);
  const search = historyEntry.substring(expectedPrefix.length);
  expect(Utils.getSearchParamsFromUrl(search)).toEqual(searchQueryParams);
}

describe('updateUrlWithViewState', () => {
  const emptyCategorizedUncheckedKeys = {
    [COLUMN_TYPES.ATTRIBUTES]: [''],
    [COLUMN_TYPES.PARAMS]: [''],
    [COLUMN_TYPES.METRICS]: [''],
    [COLUMN_TYPES.TAGS]: [''],
  };
  const defaultParameters = {
    searchInput: '',
    lifecycleFilter: DEFAULT_LIFECYCLE_FILTER,
    modelVersionFilter: DEFAULT_MODEL_VERSION_FILTER,
    orderByKey: DEFAULT_ORDER_BY_KEY,
    orderByAsc: DEFAULT_ORDER_BY_ASC,
    startTime: DEFAULT_START_TIME,
    showMultiColumns: DEFAULT_SHOW_MULTI_COLUMNS,
    categorizedUncheckedKeys: DEFAULT_CATEGORIZED_UNCHECKED_KEYS,
    diffSwitchSelected: DEFAULT_DIFF_SWITCH_SELECTED,
    preSwitchCategorizedUncheckedKeys: DEFAULT_CATEGORIZED_UNCHECKED_KEYS,
    postSwitchCategorizedUncheckedKeys: DEFAULT_CATEGORIZED_UNCHECKED_KEYS,
  };
  let wrapper;
  let instance;
  beforeEach(() => {
    localStorage.clear();
    wrapper = getExperimentPageMock({
      history: history,
    });
    instance = wrapper.instance();
  });

  test('updateUrlWithViewState updates URL correctly with default params', () => {
    const {
      searchInput,
      orderByKey,
      orderByAsc,
      startTime,
      lifecycleFilter,
      modelVersionFilter,
      showMultiColumns,
      diffSwitchSelected,
    } = defaultParameters;

    instance.updateUrlWithViewState();

    expectSearchState(history.push.mock.calls[0][0], {
      searchInput,
      orderByKey,
      orderByAsc,
      startTime,
      lifecycleFilter,
      modelVersionFilter,
      showMultiColumns,
      categorizedUncheckedKeys: emptyCategorizedUncheckedKeys,
      diffSwitchSelected,
      preSwitchCategorizedUncheckedKeys: emptyCategorizedUncheckedKeys,
      postSwitchCategorizedUncheckedKeys: emptyCategorizedUncheckedKeys,
    });
  });

  test('updateUrlWithViewState updates URL correctly with orderByAsc true', () => {
    instance.setState({
      persistedState: new ExperimentPagePersistedState({
        orderByAsc: true,
      }).toJSON(),
    });

    const {
      searchInput,
      orderByKey,
      startTime,
      lifecycleFilter,
      modelVersionFilter,
      showMultiColumns,
      diffSwitchSelected,
    } = defaultParameters;

    instance.updateUrlWithViewState();

    expectSearchState(history.push.mock.calls[0][0], {
      searchInput,
      orderByKey,
      orderByAsc: true,
      startTime,
      lifecycleFilter,
      modelVersionFilter,
      showMultiColumns,
      categorizedUncheckedKeys: emptyCategorizedUncheckedKeys,
      diffSwitchSelected,
      preSwitchCategorizedUncheckedKeys: emptyCategorizedUncheckedKeys,
      postSwitchCategorizedUncheckedKeys: emptyCategorizedUncheckedKeys,
    });
  });

  test('updateUrlWithViewState updates URL correctly with lifecycle & model filter', () => {
    instance.setState({
      persistedState: new ExperimentPagePersistedState({
        lifecycleFilter: 'life',
        modelVersionFilter: 'model',
      }).toJSON(),
    });

    const {
      searchInput,
      orderByKey,
      orderByAsc,
      startTime,
      showMultiColumns,
      diffSwitchSelected,
    } = defaultParameters;

    instance.updateUrlWithViewState();

    expectSearchState(history.push.mock.calls[0][0], {
      searchInput,
      orderByKey,
      orderByAsc,
      startTime,
      lifecycleFilter: 'life',
      modelVersionFilter: 'model',
      showMultiColumns,
      categorizedUncheckedKeys: emptyCategorizedUncheckedKeys,
      diffSwitchSelected,
      preSwitchCategorizedUncheckedKeys: emptyCategorizedUncheckedKeys,
      postSwitchCategorizedUncheckedKeys: emptyCategorizedUncheckedKeys,
    });
  });

  test('updateUrlWithViewState updates URL correctly with searchInput', () => {
    instance.setState({
      persistedState: new ExperimentPagePersistedState({
        searchInput: 'search-value',
      }).toJSON(),
    });

    const {
      orderByKey,
      orderByAsc,
      startTime,
      lifecycleFilter,
      modelVersionFilter,
      showMultiColumns,
      diffSwitchSelected,
    } = defaultParameters;

    instance.updateUrlWithViewState();

    expectSearchState(history.push.mock.calls[0][0], {
      searchInput: 'search-value',
      orderByKey,
      orderByAsc,
      startTime,
      lifecycleFilter,
      modelVersionFilter,
      showMultiColumns,
      categorizedUncheckedKeys: emptyCategorizedUncheckedKeys,
      diffSwitchSelected,
      preSwitchCategorizedUncheckedKeys: emptyCategorizedUncheckedKeys,
      postSwitchCategorizedUncheckedKeys: emptyCategorizedUncheckedKeys,
    });
  });

  test('updateUrlWithViewState updates URL correctly with showMultiColumns false', () => {
    instance.setState({
      persistedState: new ExperimentPagePersistedState({
        showMultiColumns: false,
      }).toJSON(),
    });

    const {
      searchInput,
      orderByKey,
      orderByAsc,
      startTime,
      lifecycleFilter,
      modelVersionFilter,
      diffSwitchSelected,
    } = defaultParameters;

    instance.updateUrlWithViewState();

    expectSearchState(history.push.mock.calls[0][0], {
      searchInput,
      orderByKey,
      orderByAsc,
      startTime,
      lifecycleFilter,
      modelVersionFilter,
      showMultiColumns: false,
      categorizedUncheckedKeys: emptyCategorizedUncheckedKeys,
      diffSwitchSelected,
      preSwitchCategorizedUncheckedKeys: emptyCategorizedUncheckedKeys,
      postSwitchCategorizedUncheckedKeys: emptyCategorizedUncheckedKeys,
    });
  });

  test('updateUrlWithViewState updates URL correctly with diffSwitchSelected true', () => {
    const categorizedUncheckedKeys = {
      [COLUMN_TYPES.ATTRIBUTES]: ['a1'],
      [COLUMN_TYPES.PARAMS]: ['p1'],
      [COLUMN_TYPES.METRICS]: ['m1'],
      [COLUMN_TYPES.TAGS]: ['t1'],
    };

    const preSwitchCategorizedUncheckedKeys = {
      [COLUMN_TYPES.ATTRIBUTES]: ['a2'],
      [COLUMN_TYPES.PARAMS]: ['p2'],
      [COLUMN_TYPES.METRICS]: ['m2'],
      [COLUMN_TYPES.TAGS]: ['t2'],
    };

    const postSwitchCategorizedUncheckedKeys = {
      [COLUMN_TYPES.ATTRIBUTES]: ['a3'],
      [COLUMN_TYPES.PARAMS]: ['p3'],
      [COLUMN_TYPES.METRICS]: ['m3'],
      [COLUMN_TYPES.TAGS]: ['t3'],
    };

    instance.setState({
      persistedState: new ExperimentPagePersistedState({
        diffSwitchSelected: true,
        categorizedUncheckedKeys: categorizedUncheckedKeys,
        preSwitchCategorizedUncheckedKeys: preSwitchCategorizedUncheckedKeys,
        postSwitchCategorizedUncheckedKeys: postSwitchCategorizedUncheckedKeys,
      }).toJSON(),
    });

    const {
      searchInput,
      orderByKey,
      orderByAsc,
      startTime,
      lifecycleFilter,
      modelVersionFilter,
      showMultiColumns,
    } = defaultParameters;

    instance.updateUrlWithViewState();

    expectSearchState(history.push.mock.calls[0][0], {
      searchInput,
      orderByKey,
      orderByAsc,
      startTime,
      lifecycleFilter,
      modelVersionFilter,
      showMultiColumns,
      categorizedUncheckedKeys: categorizedUncheckedKeys,
      diffSwitchSelected: true,
      preSwitchCategorizedUncheckedKeys: preSwitchCategorizedUncheckedKeys,
      postSwitchCategorizedUncheckedKeys: postSwitchCategorizedUncheckedKeys,
    });
  });
});

describe('filtersDidUpdate', () => {
  let wrapper;
  let instance;
  let prevState;
  beforeEach(() => {
    localStorage.clear();
    wrapper = getExperimentPageMock();
    instance = wrapper.instance();
    prevState = {
      persistedState: {
        searchInput: '',
        orderByKey: DEFAULT_ORDER_BY_KEY,
        orderByAsc: DEFAULT_ORDER_BY_ASC,
        startTime: DEFAULT_START_TIME,
        lifecycleFilter: DEFAULT_LIFECYCLE_FILTER,
        modelVersionFilter: DEFAULT_MODEL_VERSION_FILTER,
      },
    };
  });
  test('filtersDidUpdate returns true when filters were not updated', () => {
    expect(instance.filtersDidUpdate(prevState)).toEqual(false);
  });

  test('filtersDidUpdate returns false when searchinput was updated', () => {
    prevState.persistedState.searchInput = 'updated';
    expect(instance.filtersDidUpdate(prevState)).toEqual(true);
  });

  test('filtersDidUpdate returns false when orderByKey was updated', () => {
    prevState.persistedState.orderByKey = 'updated';
    expect(instance.filtersDidUpdate(prevState)).toEqual(true);
  });

  test('filtersDidUpdate returns false when orderByAsc was updated', () => {
    prevState.persistedState.orderByAsc = 'updated';
    expect(instance.filtersDidUpdate(prevState)).toEqual(true);
  });

  test('filtersDidUpdate returns false when startTime was updated', () => {
    prevState.persistedState.startTime = 'updated';
    expect(instance.filtersDidUpdate(prevState)).toEqual(true);
  });

  test('filtersDidUpdate returns false when lifecycleFilter was updated', () => {
    prevState.persistedState.lifecycleFilter = 'updated';
    expect(instance.filtersDidUpdate(prevState)).toEqual(true);
  });

  test('filtersDidUpdate returns false when modelVersionFilter was updated', () => {
    prevState.persistedState.modelVersionFilter = 'updated';
    expect(instance.filtersDidUpdate(prevState)).toEqual(true);
  });
});

describe('setShowMultiColumns', () => {
  test('setShowMultiColumns sets state correctly', () => {
    const wrapper = getExperimentPageMock();
    const instance = wrapper.instance();
    const updateUrlWithViewStateSpy = jest.fn();
    const snapshotComponentStateSpy = jest.fn();
    instance.updateUrlWithViewState = updateUrlWithViewStateSpy;
    instance.snapshotComponentState = snapshotComponentStateSpy;
    instance.setShowMultiColumns(true);
    expect(instance.state.persistedState.showMultiColumns).toEqual(true);
    expect(updateUrlWithViewStateSpy).toHaveBeenCalledTimes(1);
    expect(snapshotComponentStateSpy).toHaveBeenCalledTimes(1);
    instance.setShowMultiColumns(false);
    expect(instance.state.persistedState.showMultiColumns).toEqual(false);
    expect(updateUrlWithViewStateSpy).toHaveBeenCalledTimes(2);
    expect(snapshotComponentStateSpy).toHaveBeenCalledTimes(2);
  });
});

describe('handleColumnSelectionCheck', () => {
  test('handleColumnSelectionCheck sets state correctly', () => {
    const wrapper = getExperimentPageMock();
    const instance = wrapper.instance();
    const updateUrlWithViewStateSpy = jest.fn();
    const snapshotComponentStateSpy = jest.fn();
    instance.updateUrlWithViewState = updateUrlWithViewStateSpy;
    instance.snapshotComponentState = snapshotComponentStateSpy;
    instance.handleColumnSelectionCheck({
      key: 'value',
    });
    expect(instance.state.persistedState.categorizedUncheckedKeys).toEqual({
      key: 'value',
    });
    expect(updateUrlWithViewStateSpy).toHaveBeenCalledTimes(1);
    expect(snapshotComponentStateSpy).toHaveBeenCalledTimes(1);
  });
});

describe('handleDiffSwitchChange', () => {
  test('handleDiffSwitchChange sets state correctly', () => {
    const wrapper = getExperimentPageMock();
    const instance = wrapper.instance();
    const updateUrlWithViewStateSpy = jest.fn();
    const snapshotComponentStateSpy = jest.fn();
    instance.updateUrlWithViewState = updateUrlWithViewStateSpy;
    instance.snapshotComponentState = snapshotComponentStateSpy;

    instance.handleDiffSwitchChange({
      categorizedUncheckedKeys: {
        key1: 'value1',
      },
      preSwitchCategorizedUncheckedKeys: {
        key2: 'value2',
      },
      postSwitchCategorizedUncheckedKeys: {
        key3: 'value3',
      },
    });

    expect(instance.state.persistedState.categorizedUncheckedKeys).toEqual({
      key1: 'value1',
    });
    expect(instance.state.persistedState.preSwitchCategorizedUncheckedKeys).toEqual({
      key2: 'value2',
    });
    expect(instance.state.persistedState.postSwitchCategorizedUncheckedKeys).toEqual({
      key3: 'value3',
    });
    expect(instance.state.persistedState.diffSwitchSelected).toEqual(true);
    expect(updateUrlWithViewStateSpy).toHaveBeenCalledTimes(1);
    expect(snapshotComponentStateSpy).toHaveBeenCalledTimes(1);

    instance.handleDiffSwitchChange({
      categorizedUncheckedKeys: {
        key4: 'value4',
      },
    });

    expect(instance.state.persistedState.categorizedUncheckedKeys).toEqual({
      key4: 'value4',
    });
    expect(instance.state.persistedState.preSwitchCategorizedUncheckedKeys).toEqual({
      key2: 'value2',
    });
    expect(instance.state.persistedState.postSwitchCategorizedUncheckedKeys).toEqual({
      key3: 'value3',
    });
    expect(instance.state.persistedState.diffSwitchSelected).toEqual(false);
    expect(updateUrlWithViewStateSpy).toHaveBeenCalledTimes(2);
    expect(snapshotComponentStateSpy).toHaveBeenCalledTimes(2);
  });
});
