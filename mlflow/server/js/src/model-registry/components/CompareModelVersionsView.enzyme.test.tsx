/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { CompareModelVersionsView, CompareModelVersionsViewImpl } from './CompareModelVersionsView';
import React from 'react';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import configureStore from 'redux-mock-store';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';

describe('unconnected tests', () => {
  let wrapper;
  let minimumProps: any;
  let minimalStore: any;
  let commonProps: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);
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
    minimalStore = mockStore({
      entities: {
        runInfosByUuid: { 123: { dummy_key: 'dummy_value' } },
        latestMetricsByRunUuid: {
          123: [{ key: 'test_metric', value: 0.0 }],
        },
        paramsByRunUuid: { 123: [{ key: 'test_param', value: '0.0' }] },
        tagsByRunUuid: { 123: [{ key: 'test_tag', value: 'test.user' }] },
        mlModelArtifactByModelVersion: {},
      },
      apis: {},
    });
    commonProps = {
      ...minimumProps,
      runInfos: [
        {
          runUuid: '123',
          experimentId: '0',
          userId: 'test.user',
          status: 'FINISHED',
          startTime: '0',
          endTime: '21',
          artifactUri: './mlruns',
          lifecycleStage: 'active',
        },
        {
          runUuid: 'somebadrunID',
        },
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
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <CompareModelVersionsView {...minimumProps} />
        </MemoryRouter>
      </Provider>,
    );
    expect(wrapper.length).toBe(1);
  });
  test('check that the component renders correctly with common props', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <CompareModelVersionsView {...commonProps} />
        </MemoryRouter>
      </Provider>,
    );
    // Checking the breadcrumb renders correctly
    expect(wrapper.containsAllMatchingElements(['Registered Models', 'test', 'Comparing 2 Versions'])).toEqual(true);
    // Checking the model version shows up
    expect(wrapper.containsAllMatchingElements(['Model Version:', '1'])).toEqual(true);
    expect(wrapper.containsAllMatchingElements(['Model Version:', '2'])).toEqual(true);
  });
});

describe('connected tests', () => {
  let wrapper;
  let minimumProps: any;
  let minimalStore: any;
  let commonStore: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);
  beforeEach(() => {
    minimumProps = {
      modelName: 'test',
      versionsToRuns: { 1: '123' },
    };
    minimalStore = mockStore({
      entities: {
        runInfosByUuid: { 123: { dummy_key: 'dummy_value' } },
        latestMetricsByRunUuid: {
          123: [{ key: 'test_metric', value: 0.0 }],
        },
        paramsByRunUuid: { 123: [{ key: 'test_param', value: '0.0' }] },
        tagsByRunUuid: { 123: [{ key: 'test_tag', value: 'test.user' }] },
        mlModelArtifactByModelVersion: {},
      },
      apis: {},
    });
    commonStore = mockStore({
      entities: {
        runInfosByUuid: {
          123: {
            runUuid: '123',
            experimentId: '0',
            userId: 'test.user',
            status: 'FINISHED',
            startTime: '0',
            endTime: '21',
            artifactUri: './mlruns',
            lifecycleStage: 'active',
          },
        },
        latestMetricsByRunUuid: {
          123: [{ key: 'test_metric', value: 0.0, timestamp: '321', step: '42' }],
        },
        paramsByRunUuid: { 123: [{ key: 'test_param', value: '0.0' }] },
        tagsByRunUuid: { 123: [{ key: 'test_tag', value: 'test.user' }] },
        mlModelArtifactByModelVersion: {
          test: {
            1: {
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
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <CompareModelVersionsView {...minimumProps} />
        </MemoryRouter>
      </Provider>,
    );
    expect(wrapper.find(CompareModelVersionsView).length).toBe(1);
  });
  test('connected should render with minimal props and common store correctly', () => {
    wrapper = mountWithIntl(
      <Provider store={commonStore}>
        <MemoryRouter>
          <CompareModelVersionsView {...minimumProps} />
        </MemoryRouter>
      </Provider>,
    );
    expect(wrapper.find(CompareModelVersionsView).length).toBe(1);
    // Checking the breadcrumb renders correctly
    expect(wrapper.containsAllMatchingElements(['Registered Models', 'test', 'Comparing 1 Versions'])).toEqual(true);
    // Checking the model version shows up
    expect(wrapper.containsAllMatchingElements(['Model Version:', '1'])).toEqual(true);
  });
  test('validate that comparison works with null run IDs or invalid run IDs', () => {
    const testProps = {
      modelName: 'test',
      versionsToRuns: { 1: '123', 2: null, 3: 'cats' },
    };
    wrapper = mountWithIntl(
      <Provider store={commonStore}>
        <MemoryRouter>
          <CompareModelVersionsView {...testProps} />
        </MemoryRouter>
      </Provider>,
    );
    expect(wrapper.find(CompareModelVersionsView).length).toBe(1);
    // Checking the breadcrumb renders correctly
    expect(wrapper.containsAllMatchingElements(['Registered Models', 'test', 'Comparing 3 Versions'])).toEqual(true);
    // Checking the model version shows up
    expect(wrapper.containsAllMatchingElements(['Model Version:', '1'])).toEqual(true);
  });
  test('inputsList and outputsList props contains correct columns', () => {
    wrapper = mountWithIntl(
      <Provider store={commonStore}>
        <MemoryRouter>
          <CompareModelVersionsView {...minimumProps} />
        </MemoryRouter>
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
