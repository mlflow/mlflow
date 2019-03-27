import React from 'react';
import { shallow } from 'enzyme';
import { ExperimentView } from './ExperimentView';
import KeyFilter from "../utils/KeyFilter";
import {LIFECYCLE_FILTER} from "./ExperimentPage";
import Fixtures from '../test-utils/Fixtures';

const getExperimentViewMock = () => {
  return shallow(<ExperimentView
    onSearch={() => {}}
    runInfos={[]}
    experiment={Fixtures.experiments[0]}
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
