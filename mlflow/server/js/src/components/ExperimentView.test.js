import React from 'react';
import { shallow } from 'enzyme';
import { ExperimentView, mapStateToProps } from './ExperimentView';
import Fixtures from "../test-utils/Fixtures";
import {LIFECYCLE_FILTER} from "./ExperimentPage";
import KeyFilter from "../utils/KeyFilter";
import {
  addApiToState,
  addExperimentToState,
  addExperimentTagsToState,
  createPendingApi,
  emptyState} from "../test-utils/ReduxStoreFixtures";
import {getUUID} from "../Actions";
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
    experimentTags={{}}
    paramKeyFilter={new KeyFilter("")}
    metricKeyFilter={new KeyFilter("")}
    lifecycleFilter={LIFECYCLE_FILTER.ACTIVE}
    searchInput={""}
    searchRunsError={''}
    isLoading
    loadingMore={false}
    handleLoadMoreRuns={jest.fn()}
    orderByKey={null}
    orderByAsc={false}
  />);
};

test('Expanding and collapsing a parent run updates component state', () => {
  const wrapper = getExperimentViewMock();
  wrapper.instance().onExpand(Fixtures.sortedRunIds[0], Fixtures.childRunIds);
  expect(wrapper.state('persistedState')['runsExpanded']).toEqual(
    {[Fixtures.sortedRunIds[0]]: true});
  wrapper.instance().onExpand(Fixtures.sortedRunIds[0], Fixtures.childRunIds);
  expect(wrapper.state('persistedState')['runsExpanded']).toEqual(
    {[Fixtures.sortedRunIds[0]]: false});
});

test('Clicking on run checkbox selects/deselects the run and updates lastCheckboxIndex', () => {
  const wrapper = getExperimentViewMock();
  expect(wrapper.state('runsSelected')).toEqual({});
  expect(wrapper.state('lastCheckboxIndex')).not.toEqual(1);
  wrapper.instance().onCheckbox({}, Fixtures.childRunIds, 1, Fixtures.sortedRunIds);
  wrapper.update();
  expect(wrapper.state('lastCheckboxIndex')).toEqual(1);
  expect(wrapper.state('runsSelected')).toEqual({[Fixtures.runInfos[1].run_uuid]: true});
  wrapper.instance().onCheckbox({}, Fixtures.childRunIds, 1, Fixtures.sortedRunIds);
  expect(wrapper.state('runsSelected')).toEqual({});
});

test('Command or ctrl-clicking a parent run selects or deselects all child runs', () => {
  const wrapper = getExperimentViewMock();
  const clickEvents = [{metaKey: true}, {ctrlKey: true}];
  clickEvents.forEach((event) => {
    wrapper.instance().onCheckbox(event, Fixtures.childRunIds, 0, Fixtures.sortedRunIds);
    wrapper.update();
    expect(wrapper.state('runsSelected')).toEqual({
      'parent-run-id': true,
      'child-run-id-0': true,
      'child-run-id-1': true,
      'child-run-id-2': true,
    });
    wrapper.instance().onCheckbox({metaKey: true}, Fixtures.childRunIds, 0, Fixtures.sortedRunIds);
    expect(wrapper.state('runsSelected')).toEqual({});
  });
});

test('Shift-clicking across runs selects multiple runs: top level runs', () => {
  const wrapper = getExperimentViewMock();
  wrapper.instance().onCheckbox({}, [], 0, Fixtures.topLevelRunIds);
  expect(wrapper.state('runsSelected')).toEqual({[Fixtures.topLevelRunIds[0]]: true});
  wrapper.instance().onCheckbox({shiftKey: true}, [], 2, Fixtures.topLevelRunIds);
  const expectedSelected = {};
  Fixtures.topLevelRunIds.slice(0, 3).forEach((runId) => expectedSelected[runId] = true);
  expect(wrapper.state('runsSelected')).toEqual(expectedSelected);
  wrapper.instance().onCheckbox({shiftKey: true}, [], 0, Fixtures.topLevelRunIds);
  expect(wrapper.state('runsSelected')).toEqual({});
});

test('Shift-clicking across runs selects multiple runs: mixed top level and child runs', () => {
  const wrapper = getExperimentViewMock();
  wrapper.instance().onCheckbox({}, [], 2, Fixtures.sortedRunIds);
  expect(wrapper.state('runsSelected')).toEqual({'child-run-id-1': true});
  wrapper.instance().onCheckbox({shiftKey: true}, [], 6, Fixtures.sortedRunIds);
  expect(wrapper.state('runsSelected')).toEqual({
    'child-run-id-1': true,
    'child-run-id-2': true,
    'top-level-childless-run-0': true,
    'top-level-childless-run-1': true,
    'top-level-childless-run-2': true,
  });
  wrapper.instance().onCheckbox({shiftKey: true}, [], 3, Fixtures.sortedRunIds);
  expect(wrapper.state('runsSelected')).toEqual({'child-run-id-1': true});
});

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
  state = addExperimentTagsToState(state, experiment.experiment_id, []);
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
    experimentTags: {},
  });
});
