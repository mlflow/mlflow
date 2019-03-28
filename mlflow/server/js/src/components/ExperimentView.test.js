import React from 'react';
import { shallow, mount } from 'enzyme';
import { ExperimentView } from './ExperimentView';
import ExperimentRunsTableCompactView from './ExperimentRunsTableCompactView';
import KeyFilter from "../utils/KeyFilter";
import {LIFECYCLE_FILTER} from "./ExperimentPage";
import Fixtures from '../test-utils/Fixtures';
import ExperimentViewUtil from "./ExperimentViewUtil";

const getExperimentViewMock = () => {
  return mount(<ExperimentView
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

test('Child runs are unexpanded by default', () => {
  const wrapper = getExperimentViewMock();
  const tableView = wrapper.find(ExperimentRunsTableCompactView); //.dive();
  expect(tableView).toHaveLength(1);
  // expect(tableView.find(".runs-table-flex-container tr")).toHaveLength(2);
  expect(tableView.find(".runs-table-flex-container .ExperimentView-expander")).toHaveLength(2);

});

test('Clicking on run checkbox selects or deselects the run', () => {
  // const wrapper = getExperimentViewMock();
  // const tableView = wrapper.find(ExperimentRunsTableCompactView).dive();
});

test('Command-clicking a parent run selects or deselects all child runs', () => {

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
