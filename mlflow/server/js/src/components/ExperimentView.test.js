import React from 'react';
import { shallow } from 'enzyme';
import { ExperimentView, mapStateToProps } from './ExperimentView';
import Fixtures from "../test-utils/Fixtures";
import {LIFECYCLE_FILTER} from "./ExperimentPage";
import KeyFilter from "../utils/KeyFilter";
import {addApiToState, addExperimentToState, createPendingApi, emptyState} from "../test-utils/ReduxStoreFixtures";
import {getUUID, SEARCH_MAX_RESULTS} from "../Actions";
import {Spinner} from "./Spinner";

let onSearchSpy;

beforeEach(() => {
  onSearchSpy = jest.fn();
});

const getExperimentViewMock = () => {
  return shallow(<ExperimentView
    onSearch={onSearchSpy}
    runInfos={[]}
    experiment={Fixtures.createExperiment()}
    history={[]}
    paramKeyList={[]}
    metricKeyList={[]}
    paramsList={[]}
    metricsList={[]}
    tagsList={[]}
    paramKeyFilter={new KeyFilter("")}
    metricKeyFilter={new KeyFilter("")}
    lifecycleFilter={LIFECYCLE_FILTER.ACTIVE}
    searchInput={""}
    searchRunsError={''}
    isLoading
  />);
};

test('Entering filter input updates component state', () => {
  const wrapper = getExperimentViewMock();
  wrapper.instance().setState = jest.fn();
  // Test entering param filter input
  wrapper.find('.ExperimentView-paramKeyFilter input').first().simulate(
    'change', {target: {value: 'param name'}});
  expect(wrapper.instance().setState).toBeCalledWith({paramKeyFilterInput: 'param name'});
  // Test entering metric filter input
  wrapper.find('.ExperimentView-metricKeyFilter input').first().simulate(
    'change', {target: {value: 'metric name'}});
  expect(wrapper.instance().setState).toBeCalledWith({metricKeyFilterInput: 'metric name'});
  // Test entering search input
  wrapper.find('.ExperimentView-search-input input').first().simulate(
    'change', {target: {value: 'search input string'}});
  expect(wrapper.instance().setState).toBeCalledWith({searchInput: 'search input string'});
});

test("ExperimentView will show spinner if isLoading prop is true", () => {
  const wrapper = getExperimentViewMock();
  expect(wrapper.find(Spinner)).toHaveLength(1);
});

// mapStateToProps should only be run after the call to getExperiment from ExperimentPage is
// resolved
test("mapStateToProps doesn't blow up if the searchRunsApi is pending", () => {
  const searchRunsId = getUUID();
  let state = emptyState;
  const experiment = Fixtures.createExperiment();
  state = addApiToState(state, createPendingApi(searchRunsId));
  state = addExperimentToState(state, experiment);
  const newProps = mapStateToProps(state, {
    lifecycleFilter: LIFECYCLE_FILTER.ACTIVE,
    searchRunsRequestId: searchRunsId,
    experimentId: experiment.experiment_id
  });
  expect(newProps).toEqual({
    runInfos: [],
    experiment,
    metricKeyList: [],
    paramKeyList: [],
    metricsList: [],
    paramsList: [],
    tagsList: [],
  });
});

test(`Says that the SEARCH_RUNS_LIMIT is hit when more than ${SEARCH_MAX_RESULTS} runs are returned`, () => {
  const wrapper = getExperimentViewMock();
  wrapper.setProps({ runInfos: Array(SEARCH_MAX_RESULTS + 1) });
  expect(wrapper.find('.run-count').text()).toEqual(`Showing the latest ${SEARCH_MAX_RESULTS} matching runs`);
});

test(`Doesn't say the SEARCH_RUNS_LIMIT is hit when <= than ${SEARCH_MAX_RESULTS} runs are returned`, () => {
  const wrapper = getExperimentViewMock();
  wrapper.setProps({ runInfos: Array(SEARCH_MAX_RESULTS) });
  expect(wrapper.find('.run-count').text()).toEqual(`${SEARCH_MAX_RESULTS} matching runs`);
});
