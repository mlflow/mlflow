import React from 'react';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import FileSaver from 'file-saver';
import { BrowserRouter } from 'react-router-dom';

import { ExperimentViewWithIntl, mapStateToProps } from './ExperimentView';
import Fixtures from '../utils/test-utils/Fixtures';
import { LIFECYCLE_FILTER, MODEL_VERSION_FILTER } from './ExperimentPage';
import { ColumnTypes } from '../constants';
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
import { Metric, Param, RunTag, RunInfo } from '../sdk/MlflowMessages';
import { mountWithIntl, shallowWithInjectIntl } from '../../common/utils/TestUtils';

let onSearchSpy;

beforeEach(() => {
  onSearchSpy = jest.fn();
});

const getDefaultExperimentViewProps = () => {
  return {
    onSearch: onSearchSpy,
    runInfos: [
      RunInfo.fromJs({
        run_uuid: 'run-id',
        experiment_id: '3',
        status: 'FINISHED',
        start_time: 1,
        end_time: 1,
        artifact_uri: 'dummypath',
        lifecycle_stage: 'active',
      }),
    ],
    experiment: Fixtures.createExperiment(),
    history: [],
    paramKeyList: ['batch_size'],
    metricKeyList: ['acc'],
    paramsList: [[Param.fromJs({ key: 'batch_size', value: '512' })]],
    metricsList: [[Metric.fromJs({ key: 'acc', value: 0.1 })]],
    tagsList: [],
    experimentTags: {},
    paramKeyFilter: new KeyFilter(''),
    metricKeyFilter: new KeyFilter(''),
    modelVersionFilter: MODEL_VERSION_FILTER.ALL_RUNS,
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
    modelVersionsByRunUuid: {},
  };
};

const getExperimentViewMock = (componentProps = {}) => {
  const mergedProps = { ...getDefaultExperimentViewProps(), ...componentProps };
  return shallowWithInjectIntl(<ExperimentViewWithIntl {...mergedProps} />);
};

const mountExperimentViewMock = (componentProps = {}) => {
  const mergedProps = { ...getDefaultExperimentViewProps(), ...componentProps };
  const store = configureStore()(emptyState);
  return mountWithIntl(
    <Provider store={store}>
      <BrowserRouter>
        <ExperimentViewWithIntl {...mergedProps} />
      </BrowserRouter>
    </Provider>,
  );
};

test('Should render with minimal props without exploding', () => {
  const wrapper = getExperimentViewMock();
  expect(wrapper.length).toBe(1);
});

test('Should render compact view without exploding', () => {
  const wrapper = mountExperimentViewMock({ isLoading: false, forceCompactTableView: true });
  expect(wrapper.find('ExperimentRunsTableCompactView').text()).toContain('batch_size:512');
  expect(wrapper.length).toBe(1);
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
    metricKeyList: [],
    paramKeyList: [],
    metricsList: [],
    paramsList: [],
    tagsList: [],
    experimentTags: {},
    modelVersionsByRunUuid: {},
  });
});

describe('Download CSV', () => {
  const mlflowSystemTags = {
    'mlflow.runName': 'name',
    'mlflow.source.name': 'src.py',
    'mlflow.source.type': 'LOCAL',
    'mlflow.user': 'user',
  };

  const createTags = (tags) => {
    // Converts {key: value, ...} to {key: RunTag(key, value), ...}
    return Object.entries(tags).reduce(
      (acc, [key, value]) => ({ ...acc, [key]: RunTag.fromJs({ key, value }) }),
      {},
    );
  };

  const blobOptionExpected = { type: 'application/csv;charset=utf-8' };
  const filenameExpected = 'runs.csv';
  const saveAsSpy = jest.spyOn(FileSaver, 'saveAs');

  afterEach(() => {
    saveAsSpy.mockClear();
  });

  test('Downloaded CSV contains tags', () => {
    const tagsList = [
      createTags({
        ...mlflowSystemTags,
        a: '0',
        b: '1',
      }),
    ];
    const csvExpected = `
Run ID,Name,Source Type,Source Name,User,Status,batch_size,acc,a,b
run-id,name,LOCAL,src.py,user,FINISHED,512,0.1,0,1
`.substring(1); // strip a leading newline

    const wrapper = getExperimentViewMock({ tagsList });
    wrapper.instance().onDownloadCsv();
    expect(saveAsSpy).toHaveBeenCalledWith(
      new Blob([csvExpected], blobOptionExpected),
      filenameExpected,
    );
  });

  test('Downloaded CSV does not contain unchecked tags', () => {
    const tagsList = [
      createTags({
        ...mlflowSystemTags,
        a: '0',
        b: '1',
      }),
    ];
    const csvExpected = `
Run ID,Name,Source Type,Source Name,User,Status,batch_size,acc,a
run-id,name,LOCAL,src.py,user,FINISHED,512,0.1,0
`.substring(1);

    const wrapper = getExperimentViewMock({ tagsList });
    // Uncheck the tag 'b'
    wrapper.setState({
      persistedState: {
        categorizedUncheckedKeys: { [ColumnTypes.TAGS]: ['b'] },
      },
    });
    // Then, download CSV
    wrapper.instance().onDownloadCsv();
    expect(saveAsSpy).toHaveBeenCalledWith(
      new Blob([csvExpected], blobOptionExpected),
      filenameExpected,
    );
  });

  test('CSV download succeeds without tags', () => {
    const tagsList = [createTags(mlflowSystemTags)];
    const csvExpected = `
Run ID,Name,Source Type,Source Name,User,Status,batch_size,acc
run-id,name,LOCAL,src.py,user,FINISHED,512,0.1
`.substring(1);

    const wrapper = getExperimentViewMock({ tagsList });
    wrapper.instance().onDownloadCsv();
    expect(saveAsSpy).toHaveBeenCalledWith(
      new Blob([csvExpected], blobOptionExpected),
      filenameExpected,
    );
  });
});

describe('ExperimentView event handlers', () => {
  let wrapper;
  let instance;

  const getSearchParams = ({
    paramKeyFilterInput = '',
    metricKeyFilterInput = '',
    searchInput = '',
    lifecycleFilterInput = LIFECYCLE_FILTER.ACTIVE,
    modelVersionFilterInput = MODEL_VERSION_FILTER.ALL_RUNS,
    orderByKey = null,
    orderByAsc = false,
  } = {}) => [
    paramKeyFilterInput,
    metricKeyFilterInput,
    searchInput,
    lifecycleFilterInput,
    orderByKey,
    orderByAsc,
    modelVersionFilterInput,
  ];

  beforeEach(() => {
    wrapper = getExperimentViewMock({});
    instance = wrapper.instance();
  });

  test('handleLifecycleFilterInput calls onSearch with the right params', () => {
    const newFilterInput = LIFECYCLE_FILTER.DELETED;
    instance.handleLifecycleFilterInput({ key: newFilterInput });

    expect(onSearchSpy).toHaveBeenCalledTimes(1);
    expect(onSearchSpy).toBeCalledWith(
      ...getSearchParams({
        lifecycleFilterInput: newFilterInput,
      }),
    );
  });

  test('handleModelVersionFilterInput calls onSearch with the right params', () => {
    const newFilterInput = MODEL_VERSION_FILTER.WTIHOUT_MODEL_VERSIONS;
    instance.handleModelVersionFilterInput({ key: newFilterInput });

    expect(onSearchSpy).toHaveBeenCalledTimes(1);
    expect(onSearchSpy).toBeCalledWith(
      ...getSearchParams({
        modelVersionFilterInput: newFilterInput,
      }),
    );
  });

  test('onClear clears all parameters', () => {
    wrapper = getExperimentViewMock({
      lifecycleFilter: LIFECYCLE_FILTER.DELETED,
      modelVersionFilter: MODEL_VERSION_FILTER.WITH_MODEL_VERSIONS,
      searchInput: 'previous-testing',
    });
    instance = wrapper.instance();
    const testingString = 'testing';
    instance.setState({ searchInput: testingString });

    expect(wrapper.state('searchInput')).toEqual(testingString);

    instance.onClear();
    expect(onSearchSpy).toHaveBeenCalledTimes(1);
    expect(onSearchSpy).toBeCalledWith(
      ...getSearchParams({
        orderByAsc: true,
      }),
    );
  });

  test('search filters are correctly applied', () => {
    instance.onSearchInput({
      target: {
        value: 'SearchString',
      },
    });

    instance.onSortBy('orderByKey', true);

    expect(onSearchSpy).toHaveBeenCalledTimes(1);
    expect(onSearchSpy).toBeCalledWith(
      ...getSearchParams({
        orderByKey: 'orderByKey',
        orderByAsc: true,
      }),
    );

    instance.onSearch(undefined, 'SearchString');

    expect(onSearchSpy).toHaveBeenCalledTimes(2);
    expect(onSearchSpy).toBeCalledWith(
      ...getSearchParams({
        orderByKey: 'orderByKey',
        orderByAsc: true,
        mySearchInput: 'SearchString',
      }),
    );
  });
});
