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
  createPendingApi,
  emptyState,
} from '../utils/test-utils/ReduxStoreFixtures';
import Utils from '../../common/utils/Utils';
import { Spinner } from '../../common/components/Spinner';
import { ExperimentViewPersistedState } from '../sdk/MlflowLocalStorageMessages';
import { getUUID } from '../../common/utils/ActionUtils';

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
    .find('.ExperimentView-searchBox')
    .first()
    .simulate('change', { target: { value: 'search input string' } });
  expect(wrapper.instance().setState).toBeCalledWith({ searchInput: 'search input string' });
});

test('Onboarding alert shows', () => {
  const wrapper = getExperimentViewMock();
  expect(wrapper.find('Alert')).toHaveLength(1);
});

test('Onboarding alert does not show if disabled', () => {
  const wrapper = getExperimentViewMock();
  const instance = wrapper.instance();
  instance.setState({
    showOnboardingHelper: false,
  });
  expect(wrapper.find('Alert')).toHaveLength(0);
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
    tagsList: [],
    experimentTags: {},
  });
});
