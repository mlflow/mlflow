import React from 'react';
import { shallow } from 'enzyme';
import { ExperimentView } from './ExperimentView';
import KeyFilter from "../utils/KeyFilter";
import {LIFECYCLE_FILTER} from "./ExperimentPage";
import Fixtures from '../test-utils/Fixtures';


const getExperimentViewMock = () => {
  return shallow(<ExperimentView
    onSearch={() => {}}
    runInfos={Fixtures.runInfos}
    experiment={Fixtures.experiments[0]}
    history={[]}
    paramKeyList={[]}
    metricKeyList={[]}
    paramsList={Array(Fixtures.runInfos.length).fill([])}
    metricsList={Array(Fixtures.runInfos.length).fill([])}
    tagsList={Fixtures.tagsList}
    paramKeyFilter={new KeyFilter("")}
    metricKeyFilter={new KeyFilter("")}
    lifecycleFilter={LIFECYCLE_FILTER.ACTIVE}
    searchInput={""}
  />);
};

test('Entering filter input updates component state', () => {
  const wrapper = getExperimentViewMock();
  wrapper.instance().setState = jest.fn();
  // Test entering param filter input
  wrapper.find('.ExperimentView-paramKeyFilter input').first().simulate(
    'change', {target: { value: 'param name'}});
  expect(wrapper.instance().setState).toBeCalledWith({paramKeyFilterInput: 'param name'});
  // Test entering metric filter input
  wrapper.find('.ExperimentView-metricKeyFilter input').first().simulate(
    'change', {target: { value: 'metric name'}});
  expect(wrapper.instance().setState).toBeCalledWith({metricKeyFilterInput: 'metric name'});
  // Test entering search input
  wrapper.find('.ExperimentView-search-input input').first().simulate(
    'change', {target: { value: 'search input string'}});
  expect(wrapper.instance().setState).toBeCalledWith({searchInput: 'search input string'});
});

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
  wrapper.instance().onCheckbox({}, Fixtures.childRunIds, 4, Fixtures.sortedRunIds);
  expect(wrapper.state('runsSelected')).toEqual({[Fixtures.runInfos[4].run_uuid]: true});
  wrapper.instance().onCheckbox({shiftKey: true}, Fixtures.childRunIds, 6, Fixtures.sortedRunIds);
  expect(wrapper.state('runsSelected')).toEqual({
    'top-level-childless-run-0': true,
    'top-level-childless-run-1': true,
    'top-level-childless-run-2': true,
  });
  wrapper.instance().onCheckbox({shiftKey: true}, Fixtures.childRunIds, 4, Fixtures.sortedRunIds);
  expect(wrapper.state('runsSelected')).toEqual({});
});

test('Shift-clicking across runs selects multiple runs: mixed top level and child runs', () => {
  const wrapper = getExperimentViewMock();
  wrapper.instance().onCheckbox({}, Fixtures.childRunIds, 2, Fixtures.sortedRunIds);
  expect(wrapper.state('runsSelected')).toEqual({'child-run-id-1': true});
  wrapper.instance().onCheckbox({shiftKey: true}, Fixtures.childRunIds, 6, Fixtures.sortedRunIds);
  expect(wrapper.state('runsSelected')).toEqual({
    'child-run-id-1': true,
    'child-run-id-2': true,
    'top-level-childless-run-0': true,
    'top-level-childless-run-1': true,
    'top-level-childless-run-2': true,
  });
  wrapper.instance().onCheckbox({shiftKey: true}, Fixtures.childRunIds, 3, Fixtures.sortedRunIds);
  expect(wrapper.state('runsSelected')).toEqual({'child-run-id-1': true});
});

