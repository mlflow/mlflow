import React from 'react';
import { shallow, mount } from 'enzyme';
import { ExperimentView } from './ExperimentView';
import ExperimentRunsTableCompactView from './ExperimentRunsTableCompactView';
import KeyFilter from "../utils/KeyFilter";
import {LIFECYCLE_FILTER} from "./ExperimentPage";
import Fixtures from '../test-utils/Fixtures';
import ExperimentViewUtil from "./ExperimentViewUtil";
import { CellMeasurer, CellMeasurerCache, AutoSizer, Column, Table } from 'react-virtualized';
import { BrowserRouter as Router } from 'react-router-dom';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store' //ES6 modules


const getExperimentViewMock = () => {
  const initialState = {};
  const middlewares = []
  const mockStore = configureStore(middlewares)
  const store = mockStore(initialState);
  return mount(<Provider store={store}><Router><ExperimentView
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
  /></Router></Provider>);
};

const getShallowExperimentViewMock = () => {
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
  const wrapper = getShallowExperimentViewMock();
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

test('Child runs can be expanded and collapsed', () => {
  const wrapper = getExperimentViewMock();
  const tableView = wrapper.find(ExperimentRunsTableCompactView);
  expect(tableView).toHaveLength(1);
  // The test data should contain two top-level runs, one of which has three child runs
  const tableRows = tableView.find('.ReactVirtualized__Table__row');
  expect(tableRows).toHaveLength(2);
  const expanders = tableView.find('.ExperimentView-expander');
  expect(expanders).toHaveLength(1);
  expanders.simulate('click');
  wrapper.update();
  const tableRows1 = wrapper.find('.ReactVirtualized__Table__row');
  expect(tableRows1).toHaveLength(5);
  // Collapse the expanded runs
  const expanders1 = tableView.find('.ExperimentView-expander');
  expanders1.simulate('click');
  wrapper.update();
  const tableRows2 =  wrapper.find('.ReactVirtualized__Table__row');
  expect(tableRows2).toHaveLength(2);
});

test('Clicking on run checkbox selects or deselects the run', () => {
  const wrapper = getExperimentViewMock();
  const checkboxes = wrapper.find('.ReactVirtualized__Table__row input');
  checkboxes.first().simulate('click');
  console.log(wrapper.find(ExperimentView).get(0).state);
  expect(wrapper.state().runsSelected).toEqual({"abc": "def"});
  // const tableView = wrapper.find(ExperimentRunsTableCompactView).dive();
});

test('Command-clicking a parent run selects or deselects all child runs', () => {
  const wrapper = getExperimentViewMock();
  const checkboxes = wrapper.find('.ReactVirtualized__Table__row input');
  checkboxes.first().simulate('click', {metaKey: true});
});


test('Shift-clicking across runs selects multiple runs', () => {
  const wrapper = getExperimentViewMock();
  const checkboxes = wrapper.find('.ReactVirtualized__Table__row input');
  checkboxes.first().simulate('click', {metaKey: true});
});



//
// test('We correctly identify child runs and compute correct sort orderings', () => {
//   ExperimentViewUtil.getRowRenderMetadata(
//     {
//       runInfos: Fixtures.runInfos,
//       sortState: {},
//       paramsList: Fixtures.paramsList,
//       metricsList: Fixtures.metricsList,
//       tagsList: Fixtures.tagsList,
//       runsExpanded: {}
//     });
//   const wrapper = getExperimentViewMock();
//   wrapper.find('.ExperimentView-search-input input').first().simulate(
//     'change', {target: { value: 'search input string'}});
//
// });
//
