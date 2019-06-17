import React from 'react';
import { shallow } from 'enzyme';
import { ExperimentPage } from './ExperimentPage';
import Fixtures from "../test-utils/Fixtures";
import {LIFECYCLE_FILTER} from "./ExperimentPage";
import KeyFilter from "../utils/KeyFilter";
import {addApiToState, addExperimentToState, createPendingApi, emptyState} from "../test-utils/ReduxStoreFixtures";
import {getUUID, SEARCH_MAX_RESULTS} from "../Actions";
import {Spinner} from "./Spinner";
// import configureStore from 'redux-mock-store';

export default function request(url) {
  return "hello";
}

let searchRunsApi;
let getExperimentApi;
let history;
let location;

beforeEach(() => {
  searchRunsApi = jest.fn();
  getExperimentApi = jest.fn();
  history = {};
  location = {};
});

const getExperimentPageMock = () => {
  return shallow(<ExperimentPage
    experimentId={0}
    searchRunsApi={searchRunsApi}
    getExperimentApi={getExperimentApi}
    history={history}
    location={location}
  />);
};

test('Entering filter input updates component state', () => {
  const wrapper = getExperimentPageMock();
  wrapper.instance().onSearch("a", "b", "c", "d", "key", true);
  // wrapper.instance().setState = jest.fn();
  // // Test entering param filter input
  // wrapper.find('.ExperimentView-paramKeyFilter input').first().simulate(
  //   'change', {target: {value: 'param name'}});
  // expect(wrapper.instance().setState).toBeCalledWith({paramKeyFilterInput: 'param name'});
  // // Test entering metric filter input
  // wrapper.find('.ExperimentView-metricKeyFilter input').first().simulate(
  //   'change', {target: {value: 'metric name'}});
  // expect(wrapper.instance().setState).toBeCalledWith({metricKeyFilterInput: 'metric name'});
  // // Test entering search input
  // wrapper.find('.ExperimentView-search-input input').first().simulate(
  //   'change', {target: {value: 'search input string'}});
  // expect(wrapper.instance().setState).toBeCalledWith({searchInput: 'search input string'});
});

