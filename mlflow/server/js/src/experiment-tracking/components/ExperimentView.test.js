import React from 'react';
import { shallow } from 'enzyme';
import { ExperimentView, mapStateToProps } from './ExperimentView';
import Fixtures from '../utils/test-utils/Fixtures';
import { LIFECYCLE_FILTER } from './ExperimentPage';
import KeyFilter from '../utils/KeyFilter';
import {
  addApiToState,
  addExperimentToState,
  addExperimentTagsToState,
  addRunToState,
  createPendingApi,
  emptyState,
} from '../utils/test-utils/ReduxStoreFixtures';
import Utils from '../../common/utils/Utils';
import { Spinner } from '../../common/components/Spinner';
import { ExperimentViewPersistedState } from '../sdk/MlflowLocalStorageMessages';
import { getUUID } from '../../common/utils/ActionUtils';
import { RunInfo } from '../sdk/MlflowMessages';
import { ColumnTypes } from '../constants';

let onSearchSpy;

beforeEach(() => {
  onSearchSpy = jest.fn();
});

const getDefaultExperimentViewProps = () => {
  return {
    onSearch: onSearchSpy,
    runInfos: [],
    experiment: Fixtures.createExperiment(),
    history: [],
    paramKeyList: [],
    metricKeyList: [],
    paramsList: [],
    metricsList: [],
    tagsList: [],
    experimentTags: {},
    paramKeyFilter: new KeyFilter(''),
    metricKeyFilter: new KeyFilter(''),
    lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
    searchInput: '',
    searchRunsError: '',
    isLoading: true,
    loadingMore: false,
    handleLoadMoreRuns: jest.fn(),
    orderByKey: null,
    orderByAsc: false,
    setExperimentTagApi: jest.fn(),
    location: { pathname: '/' },
    tagKeyList: [],
  };
};

const getExperimentViewMock = (componentProps = {}) => {
  const mergedProps = { ...getDefaultExperimentViewProps(), ...componentProps };
  return shallow(<ExperimentView {...mergedProps} />);
};

test(`Clearing filter state calls search handler with correct arguments`, () => {
  const wrapper = getExperimentViewMock();
  wrapper.instance().onClear();
  expect(onSearchSpy.mock.calls.length).toBe(1);
  expect(onSearchSpy.mock.calls[0][0]).toBe('');
  expect(onSearchSpy.mock.calls[0][1]).toBe('');
  expect(onSearchSpy.mock.calls[0][2]).toBe('');
  expect(onSearchSpy.mock.calls[0][3]).toBe(LIFECYCLE_FILTER.ACTIVE);
  expect(onSearchSpy.mock.calls[0][4]).toBe(null);
  expect(onSearchSpy.mock.calls[0][5]).toBe(true);
});

test('Entering search input updates component state', () => {
  const wrapper = getExperimentViewMock();
  wrapper.instance().setState = jest.fn();
  // Test entering search input
  wrapper
    .find('.ExperimentView-search-input input')
    .first()
    .simulate('change', { target: { value: 'search input string' } });
  expect(wrapper.instance().setState).toBeCalledWith({ searchInput: 'search input string' });
});

test('ExperimentView will show spinner if isLoading prop is true', () => {
  const wrapper = getExperimentViewMock();
  const instance = wrapper.instance();
  instance.setState({
    persistedState: new ExperimentViewPersistedState({ showMultiColumns: false }).toJSON(),
  });
  expect(wrapper.find(Spinner)).toHaveLength(1);
});

test('Page title is set', () => {
  const mockUpdatePageTitle = jest.fn();
  Utils.updatePageTitle = mockUpdatePageTitle;
  getExperimentViewMock();
  expect(mockUpdatePageTitle.mock.calls[0][0]).toBe('Default - MLflow Experiment');
});

// mapStateToProps should only be run after the call to getExperiment from ExperimentPage is
// resolved
test("mapStateToProps doesn't blow up if the searchRunsApi is pending", () => {
  const searchRunsId = getUUID();
  let state = emptyState;
  const experiment = Fixtures.createExperiment();
  state = addApiToState(state, createPendingApi(searchRunsId));
  state = addExperimentToState(state, experiment);
  state = addExperimentTagsToState(state, experiment.experiment_id, []);
  const newProps = mapStateToProps(state, {
    lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
    searchRunsRequestId: searchRunsId,
    experimentId: experiment.experiment_id,
  });
  expect(newProps).toEqual({
    runInfos: [],
    experiment,
    metricKeyList: [],
    paramKeyList: [],
    metricsList: [],
    paramsList: [],
    tagKeyList: [],
    tagsList: [],
    experimentTags: {},
  });
});

test('params, metrics and tags computation in mapStateToProps', () => {
  const searchRunsId = getUUID();
  let state = emptyState;
  const experiment = Fixtures.createExperiment();
  const run_info = {
    run_uuid: '0',
    experiment_id: experiment.experiment_id.toString(),
    lifecycle_stage: 'active',
  };
  const run_data = {
    metrics: [
      {
        key: 'metric0',
        step: 0,
        value: 0.0,
        timestamp: 0,
      },
      {
        key: 'metric1',
        step: 0,
        value: 1.0,
        timestamp: 0,
      },
    ],
    params: [
      {
        key: 'param0',
        value: 'val0',
      },
    ],
    tags: [
      {
        key: 'tag0',
        value: 'val1',
      },
    ],
  };

  state = addRunToState(state, run_info, run_data);
  state = addExperimentToState(state, experiment);
  state = addExperimentTagsToState(state, experiment.experiment_id, []);
  const newProps = mapStateToProps(state, {
    lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
    searchRunsRequestId: searchRunsId,
    experimentId: experiment.experiment_id,
    metricKeysList: ['metric2'],
    paramKeysList: ['param1'],
    tagKeysList: ['tag1'],
  });
  expect(newProps.runInfos).toEqual([RunInfo.fromJs(run_info)]);
  expect(newProps.metricKeyList).toEqual(['metric0', 'metric1', 'metric2']);
  expect(newProps.paramKeyList).toEqual(['param0', 'param1']);
  expect(newProps.tagKeyList).toEqual(['tag0', 'tag1']);
});

test('test on filter changes call the correct !search on backend', () => {
  const wrapper = getExperimentViewMock();
  const instance = wrapper.instance();
  instance.initiateSearch = jest.fn();
  //  Set current search
  instance.state = { searchInput: 'params.foo == "bar" AND metrics.acc > 42' };
  const filters = { 'params.method': ['contains', 'adam'], 'metrics.it': ['lessThan', '0.3'] };

  instance.onFilter(filters);

  expect(instance.initiateSearch).toHaveBeenCalledWith({
    searchInput:
      'params.foo == "bar" AND metrics.acc > 42 AND params.method LIKE \'%adam%\' AND metrics.it <= 0.3',
  });
});

test('test conversion from list of columns to object of checked keys', () => {
  const wrapper = getExperimentViewMock();
  const instance = wrapper.instance();
  expect(
    instance.getCategorizedCheckedKeysFromList([
      'params.myparam',
      'metrics.mymetric',
      'tags.mytag',
      'tags.`mlflow.user`',
      'attributes.start_time',
    ]),
  ).toEqual({
    [ColumnTypes.ATTRIBUTES]: ['User', 'Start Time'],
    [ColumnTypes.METRICS]: ['mymetric'],
    [ColumnTypes.PARAMS]: ['myparam'],
    [ColumnTypes.TAGS]: ['mytag'],
  });
});

test('test bad name of columns are discarded and do not crash', () => {
  const wrapper = getExperimentViewMock();
  const instance = wrapper.instance();
  expect(
    instance.getCategorizedCheckedKeysFromList(['parameters.myparam', 'attributes.foo']),
  ).toEqual({
    [ColumnTypes.ATTRIBUTES]: [],
    [ColumnTypes.METRICS]: [],
    [ColumnTypes.PARAMS]: [],
    [ColumnTypes.TAGS]: [],
  });
});

test('test conversion categorizedKeys to list of columns', () => {
  const wrapper = getExperimentViewMock();
  const instance = wrapper.instance();
  expect(
    instance.convertCategorizedCheckedKeysToList({
      [ColumnTypes.ATTRIBUTES]: ['Start Time'],
      [ColumnTypes.METRICS]: ['m1', 'm2'],
      [ColumnTypes.PARAMS]: ['p1'],
      [ColumnTypes.TAGS]: ['t1', 't2', 't3'],
    }),
  ).toEqual([
    'attributes.start_time',
    'metrics.m1',
    'metrics.m2',
    'params.p1',
    'tags.t1',
    'tags.t2',
    'tags.t3',
  ]);
});
