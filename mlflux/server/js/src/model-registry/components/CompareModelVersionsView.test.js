import { shallow, mount } from 'enzyme';
import { CompareModelVersionsView, CompareModelVersionsViewImpl } from './CompareModelVersionsView';
import React from 'react';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import configureStore from 'redux-mock-store';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { RunInfo } from '../../experiment-tracking/sdk/MlflowMessages';

describe('unconnected tests', () => {
  let wrapper;
  let minimumProps;
  let commonProps;

  beforeEach(() => {
    minimumProps = {
      modelName: 'test',
      versionsToRuns: { 1: '123', 2: 'somebadrunID' },
      runUuids: ['123'],
      runInfos: [],
      runInfosValid: [],
      metricLists: [],
      paramLists: [],
      runNames: [],
      runDisplayNames: [],
      inputsListByName: [],
      inputsListByIndex: [],
      outputsListByName: [],
      outputsListByIndex: [],
    };

    commonProps = {
      ...minimumProps,
      runInfos: [
        RunInfo.fromJs({
          run_uuid: '123',
          experiment_id: '0',
          user_id: 'test.user',
          status: 'FINISHED',
          start_time: '0',
          end_time: '21',
          artifact_uri: './mlruns',
          lifecycle_stage: 'active',
        }),
        RunInfo.fromJs({
          run_uuid: 'somebadrunID',
        }),
      ],
      runInfosValid: [true, false],
      metricLists: [[{ key: 'test_metric', value: 0.0 }]],
      paramLists: [[{ key: 'test_param', value: '0.0' }]],
      inputsListByName: [],
      inputsListByIndex: [],
      outputsListByName: [],
      outputsListByIndex: [],
    };
  });

  test('unconnected should render with minimal props without exploding', () => {
    wrapper = shallow(<CompareModelVersionsViewImpl {...minimumProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('check that the component renders correctly with common props', () => {
    wrapper = shallow(<CompareModelVersionsViewImpl {...commonProps} />);

    // Checking the breadcrumb renders correctly
    expect(
      wrapper.containsAllMatchingElements(['Registered Models', 'test', 'Comparing 2 Versions']),
    ).toEqual(true);

    // Checking the model version shows up
    expect(wrapper.containsAllMatchingElements(['Model Version:', '1'])).toEqual(true);
    expect(wrapper.containsAllMatchingElements(['Model Version:', '2'])).toEqual(true);
  });
});

describe('connected tests', () => {
  let wrapper;
  let minimumProps;
  let minimalStore;
  let commonStore;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    minimumProps = {
      modelName: 'test',
      versionsToRuns: { 1: '123' },
    };

    minimalStore = mockStore({
      entities: {
        runInfosByUuid: { '123': RunInfo.fromJs({ dummy_key: 'dummy_value' }) },
        latestMetricsByRunUuid: {
          '123': [{ key: 'test_metric', value: 0.0 }],
        },
        paramsByRunUuid: { '123': [{ key: 'test_param', value: '0.0' }] },
        tagsByRunUuid: { '123': [{ key: 'test_tag', value: 'test.user' }] },
        mlModelArtifactByModelVersion: {},
      },
      apis: {},
    });

    commonStore = mockStore({
      entities: {
        runInfosByUuid: {
          '123': RunInfo.fromJs({
            run_uuid: '123',
            experiment_id: '0',
            user_id: 'test.user',
            status: 'FINISHED',
            start_time: '0',
            end_time: '21',
            artifact_uri: './mlruns',
            lifecycle_stage: 'active',
          }),
        },
        latestMetricsByRunUuid: {
          '123': [{ key: 'test_metric', value: 0.0, timestamp: '321', step: '42' }],
        },
        paramsByRunUuid: { '123': [{ key: 'test_param', value: '0.0' }] },
        tagsByRunUuid: { '123': [{ key: 'test_tag', value: 'test.user' }] },
        mlModelArtifactByModelVersion: {
          test: {
            '1': {
              signature: {
                inputs:
                  '[{"name": "sepal length (cm)", "type": "double"}, ' +
                  '{"name": "sepal width (cm)", "type": "double"}, ' +
                  '{"name": "petal length (cm)", "type": "double"}, ' +
                  '{"name":"petal width (cm)", "type": "double"}, ' +
                  '{"type": "double"}]',
                outputs: '[{"type": "double"}]',
              },
            },
          },
        },
      },
      apis: {},
    });
  });

  test('connected should render with minimal props and minimal store without exploding', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <CompareModelVersionsView {...minimumProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(CompareModelVersionsView).length).toBe(1);
  });

  test('connected should render with minimal props and common store correctly', () => {
    wrapper = mount(
      <Provider store={commonStore}>
        <BrowserRouter>
          <CompareModelVersionsView {...minimumProps} />
        </BrowserRouter>
      </Provider>,
    );

    expect(wrapper.find(CompareModelVersionsView).length).toBe(1);

    // Checking the breadcrumb renders correctly
    expect(
      wrapper.containsAllMatchingElements(['Registered Models', 'test', 'Comparing 1 Versions']),
    ).toEqual(true);

    // Checking the model version shows up
    expect(wrapper.containsAllMatchingElements(['Model Version:', '1'])).toEqual(true);
  });

  test('validate that comparison works with null run IDs or invalid run IDs', () => {
    const testProps = {
      modelName: 'test',
      versionsToRuns: { 1: '123', 2: null, 3: 'cats' },
    };
    wrapper = mount(
      <Provider store={commonStore}>
        <BrowserRouter>
          <CompareModelVersionsView {...testProps} />
        </BrowserRouter>
      </Provider>,
    );

    expect(wrapper.find(CompareModelVersionsView).length).toBe(1);

    // Checking the breadcrumb renders correctly
    expect(
      wrapper.containsAllMatchingElements(['Registered Models', 'test', 'Comparing 3 Versions']),
    ).toEqual(true);

    // Checking the model version shows up
    expect(wrapper.containsAllMatchingElements(['Model Version:', '1'])).toEqual(true);
  });

  test('inputsList and outputsList props contains correct columns', () => {
    wrapper = mount(
      <Provider store={commonStore}>
        <BrowserRouter>
          <CompareModelVersionsView {...minimumProps} />
        </BrowserRouter>
      </Provider>,
    );
    const props = wrapper.find(CompareModelVersionsViewImpl).props();
    expect(props.inputsListByName).toEqual([
      [
        { key: 'sepal length (cm)', value: 'double' },
        { key: 'sepal width (cm)', value: 'double' },
        { key: 'petal length (cm)', value: 'double' },
        { key: 'petal width (cm)', value: 'double' },
        { key: '-', value: 'double' },
      ],
    ]);
    expect(props.inputsListByIndex).toEqual([
      [
        { key: 0, value: 'sepal length (cm): double' },
        { key: 1, value: 'sepal width (cm): double' },
        { key: 2, value: 'petal length (cm): double' },
        { key: 3, value: 'petal width (cm): double' },
        { key: 4, value: 'double' },
      ],
    ]);
    expect(props.outputsListByName).toEqual([[{ key: '-', value: 'double' }]]);
    expect(props.outputsListByIndex).toEqual([[{ key: 0, value: 'double' }]]);
  });
});
