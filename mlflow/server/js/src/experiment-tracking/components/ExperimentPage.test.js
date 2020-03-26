import React from 'react';
import qs from 'qs';
import { shallow } from 'enzyme';
import { ErrorCodes } from '../../common/constants';
import { ExperimentPage } from './ExperimentPage';
import ExperimentView from "./ExperimentView";
import { PermissionDeniedView } from "./PermissionDeniedView";
import { ViewType } from '../sdk/MlflowEnums';
import { MemoryRouter as Router } from 'react-router-dom';
import { ErrorWrapper } from '../../common/utils/ActionUtils';


const BASE_PATH = "/experiments/17/s";
const EXPERIMENT_ID = 17;

let searchRunsApi;
let getExperimentApi;
let loadMoreRunsApi;
let history;
let location;

beforeEach(() => {
  searchRunsApi = jest.fn(() => Promise.resolve());
  getExperimentApi = jest.fn(() => Promise.resolve());
  loadMoreRunsApi = jest.fn(() => Promise.resolve());
  location = {};

  history = {};
  history.location = {};
  history.location.pathname = BASE_PATH;
  history.location.search = "";
  history.push = jest.fn();
});

const getExperimentPageMock = () => {
  return shallow(<ExperimentPage
    experimentId={EXPERIMENT_ID}
    searchRunsApi={searchRunsApi}
    getExperimentApi={getExperimentApi}
    loadMoreRunsApi={loadMoreRunsApi}
    history={history}
    location={location}
  />);
};

function expectSearchState(historyEntry, state) {
  const expectedPrefix = BASE_PATH + "?";
  expect(historyEntry.startsWith(expectedPrefix)).toBe(true);
  const search = historyEntry.substring(expectedPrefix.length);
  const parsedHistory = qs.parse(search);
  expect(parsedHistory).toEqual(state);
}

test('URL is empty for blank search', () => {
  const wrapper = getExperimentPageMock();
  wrapper.instance().onSearch("", "", "", "Active", null, true);
  expectSearchState(history.push.mock.calls[0][0], {});
  const searchRunsCall = searchRunsApi.mock.calls[1];
  expect(searchRunsCall[0]).toEqual([EXPERIMENT_ID]);
  expect(searchRunsCall[1]).toEqual("");
  expect(searchRunsCall[2]).toEqual(ViewType.ACTIVE_ONLY);
  expect(searchRunsCall[3]).toEqual([]);
});

test('URL can encode a complete search', () => {
  const wrapper = getExperimentPageMock();
  wrapper.instance().onSearch("key_filter", "metric0, metric1", "metrics.metric0 > 3",
    "Deleted", null, true);
  expectSearchState(history.push.mock.calls[0][0], {
    "metrics": "metric0, metric1",
    "params": "key_filter",
    "search": "metrics.metric0 > 3",
  });
  const searchRunsCall = searchRunsApi.mock.calls[1];
  expect(searchRunsCall[1]).toEqual("metrics.metric0 > 3");
  expect(searchRunsCall[2]).toEqual(ViewType.DELETED_ONLY);
});

test('URL can encode order_by', () => {
  const wrapper = getExperimentPageMock();
  wrapper.instance().onSearch("key_filter", "metric0, metric1", "",
    "Active", "my_key", false);
  expectSearchState(history.push.mock.calls[0][0], {
    "metrics": "metric0, metric1",
    "params": "key_filter",
    "orderByKey": "my_key",
    "orderByAsc": "false",
  });
  const searchRunsCall = searchRunsApi.mock.calls[1];
  expect(searchRunsCall[1]).toEqual("");
  expect(searchRunsCall[3]).toEqual(["my_key DESC"]);
});

test('Loading state without any URL params', () => {
  const wrapper = getExperimentPageMock();
  const state = wrapper.instance().state;
  expect(state.persistedState.paramKeyFilterString).toEqual("");
  expect(state.persistedState.metricKeyFilterString).toEqual("");
  expect(state.persistedState.searchInput).toEqual("");
  expect(state.persistedState.orderByKey).toBe(null);
  expect(state.persistedState.orderByAsc).toEqual(true);
});

test('Loading state with all URL params', () => {
  location.search = "params=a&metrics=b&search=c&orderByKey=d&orderByAsc=false";
  const wrapper = getExperimentPageMock();
  const state = wrapper.instance().state;
  expect(state.persistedState.paramKeyFilterString).toEqual("a");
  expect(state.persistedState.metricKeyFilterString).toEqual("b");
  expect(state.persistedState.searchInput).toEqual("c");
  expect(state.persistedState.orderByKey).toEqual("d");
  expect(state.persistedState.orderByAsc).toEqual(false);
});

test('should render permission denied view when getExperiment yields permission error', () => {
  const experimentPageInstance = getExperimentPageMock().instance();
  const errorMessage = "Access Denied";
  const responseErrorWrapper = new ErrorWrapper({
    responseText: `{"error_code": "${ErrorCodes.PERMISSION_DENIED}", "message": "${errorMessage}"}`,
  });
  const searchRunsErrorRequest = {
    id: experimentPageInstance.searchRunsRequestId,
    active: false,
    error: responseErrorWrapper,
  };
  const getExperimentErrorRequest = {
    id: experimentPageInstance.getExperimentRequestId,
    active: false,
    error: responseErrorWrapper,
  };
  const experimentViewInstance = shallow(experimentPageInstance.renderExperimentView(
    false,
    true,
    [searchRunsErrorRequest, getExperimentErrorRequest],
  )).instance();
  expect(experimentViewInstance).toBeInstanceOf(PermissionDeniedView);
  expect(experimentViewInstance.props.errorMessage).toEqual(errorMessage);
});

test('should render experiment view when search error occurs', () => {
  const experimentPageInstance = getExperimentPageMock().instance();
  const responseErrorWrapper = new ErrorWrapper({
    responseText: `{"error_code": "${ErrorCodes.INVALID_PARAMETER_VALUE}", "message": "Invalid"}`,
  });
  const searchRunsErrorRequest = {
    id: experimentPageInstance.searchRunsRequestId,
    active: false,
    error: responseErrorWrapper,
  };
  const getExperimentErrorRequest = {
    id: experimentPageInstance.getExperimentRequestId,
    active: false,
  };
  const renderedView = shallow(
    <Router>
      {experimentPageInstance.renderExperimentView(
      false,
      true,
      [searchRunsErrorRequest, getExperimentErrorRequest])}
    </Router>
  );
  expect(renderedView.find(ExperimentView)).toHaveLength(1);
});

test('should update next page token initially', () => {
  const promise = Promise.resolve({ value: { next_page_token: 'token_1' } });
  searchRunsApi = jest.fn(() => promise);
  const wrapper = getExperimentPageMock();
  const instance = wrapper.instance();
  return promise.then(() => expect(instance.state.nextPageToken).toBe('token_1'));
});

test('should update next page token after load-more', () => {
  const promise = Promise.resolve({ value: { next_page_token: 'token_1' } });
  loadMoreRunsApi = jest.fn(() => promise);
  const wrapper = getExperimentPageMock();
  const instance = wrapper.instance();
  instance.handleLoadMoreRuns();
  return promise.then(() => expect(instance.state.nextPageToken).toBe('token_1'));
});

test('should update next page token to null when load-more response has no token', () => {
  const promise1 = Promise.resolve({ value: { next_page_token: 'token_1' } });
  const promise2 = Promise.resolve({ value: {} });
  searchRunsApi = jest.fn(() => promise1);
  loadMoreRunsApi = jest.fn(() => promise2);
  const wrapper = getExperimentPageMock();
  const instance = wrapper.instance();
  instance.handleLoadMoreRuns();
  return Promise.all([promise1, promise2]).then(() =>
    expect(instance.state.nextPageToken).toBe(null),
  );
});
