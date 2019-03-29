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
