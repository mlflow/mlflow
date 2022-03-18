import React from 'react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { RunView, RunViewImpl } from './RunView';
import { Experiment, RunInfo, RunTag, Metric, Param } from '../sdk/MlflowMessages';
import { ArtifactNode } from '../utils/ArtifactUtils';
import { mockModelVersionDetailed } from '../../model-registry/test-utils';
import { ModelVersionStatus, Stages } from '../../model-registry/constants';
import { GET_METRIC_HISTORY_API } from '../actions';
import { mountWithIntl } from '../../common/utils/TestUtils';
import { pending } from '../../common/utils/ActionUtils';

describe('RunView', () => {
  let minimalProps;
  let minimalStoreRaw;
  let minimalStore;
  let wrapper;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    minimalProps = {
      runUuid: 'uuid-1234-5678-9012',
      experimentId: '12345',
      getMetricPagePath: jest.fn(),
      handleSetRunTag: jest.fn(),
      setTagApi: jest.fn(),
      deleteTagApi: jest.fn(),
    };
    const modelVersion = mockModelVersionDetailed(
      'Model A',
      1,
      Stages.PRODUCTION,
      ModelVersionStatus.READY,
    );
    minimalStoreRaw = {
      entities: {
        runInfosByUuid: {
          'uuid-1234-5678-9012': RunInfo.fromJs({
            run_uuid: 'uuid-1234-5678-9012',
            experiment_id: '12345',
            user_id: 'me@me.com',
            status: 'RUNNING',
            artifact_uri: 'dbfs:/databricks/abc/uuid-1234-5678-9012',
            lifecycle_stage: 'active',
          }),
        },
        artifactsByRunUuid: { 'uuid-1234-5678-9012': new ArtifactNode(true) },
        experimentsById: {
          '12345': Experiment.fromJs({
            experiment_id: '12345',
            name: 'my experiment',
            artifact_location: 'dbfs:/databricks/abc',
            lifecycle_stage: 'active',
            last_update_time: 12345678999,
            creation_time: 12345678900,
            tags: [],
          }),
        },
        modelVersionsByModel: {
          'Model A': {
            '1': modelVersion,
          },
        },
        tagsByRunUuid: { 'uuid-1234-5678-9012': {} },
        paramsByRunUuid: { 'uuid-1234-5678-9012': {} },
        latestMetricsByRunUuid: { 'uuid-1234-5678-9012': {} },
        minMetricsByRunUuid: {},
        maxMetricsByRunUuid: {},
        artifactRootUriByRunUuid: { 'uuid-1234-5678-9012': 'root/uri' },
      },
      apis: {},
    };
    minimalStore = mockStore(minimalStoreRaw);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <RunView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(RunView).length).toBe(1);
  });

  test('With no tags, params, duration - getRunCommand & metadata list', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <RunView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    ).find(RunView);

    const instance = wrapper.find(RunViewImpl).instance();
    expect(instance.getRunCommand()).toBeNull();

    expect(wrapper.html()).not.toContain('Git Commit');
    expect(wrapper.html()).not.toContain('Entry Point');
    expect(wrapper.html()).not.toContain('Duration');
    expect(wrapper.html()).not.toContain('Parent Run');
    expect(wrapper.html()).not.toContain('Job Output');
  });

  test('With non-empty tags, params, duration - getRunCommand & metadata list', () => {
    const store = mockStore({
      ...minimalStoreRaw,
      entities: {
        ...minimalStoreRaw.entities,
        runInfosByUuid: {
          ...minimalStoreRaw.entities.runInfosByUuid,
          'uuid-1234-5678-9012': RunInfo.fromJs({
            run_uuid: 'uuid-1234-5678-9012',
            experiment_id: '12345',
            user_id: 'me@me.com',
            status: 'RUNNING',
            start_time: 12345678990,
            end_time: 12345678999,
            artifact_uri: 'dbfs:/databricks/abc/uuid-1234-5678-9012',
            lifecycle_stage: 'active',
          }),
        },
        tagsByRunUuid: {
          'uuid-1234-5678-9012': {
            'mlflow.source.type': RunTag.fromJs({ key: 'mlflow.source.type', value: 'PROJECT' }),
            'mlflow.source.name': RunTag.fromJs({ key: 'mlflow.source.name', value: 'notebook' }),
            'mlflow.source.git.commit': RunTag.fromJs({
              key: 'mlflow.source.git.commit',
              value: 'abc',
            }),
            'mlflow.project.entryPoint': RunTag.fromJs({
              key: 'mlflow.project.entryPoint',
              value: 'entry',
            }),
            'mlflow.project.backend': RunTag.fromJs({
              key: 'mlflow.project.backend',
              value: 'databricks',
            }),
            'mlflow.parentRunId': RunTag.fromJs({
              key: 'mlflow.parentRunId',
              value: 'run2-5656-7878-9090',
            }),
            'mlflow.databricks.runURL': RunTag.fromJs({
              key: 'mlflow.databricks.runURL',
              value: 'https:/databricks.com/jobs_url/123',
            }),
          },
        },
        paramsByRunUuid: {
          'uuid-1234-5678-9012': {
            p1: Param.fromJs({ key: 'p1', value: 'v1' }),
            p2: Param.fromJs({ key: 'p2', value: 'v2' }),
          },
        },
      },
    });
    wrapper = mountWithIntl(
      <Provider store={store}>
        <BrowserRouter>
          <RunView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    ).find(RunView);

    const instance = wrapper.find(RunViewImpl).instance();
    expect(instance.getRunCommand()).toEqual(
      'mlflow run notebook -v abc -e entry -b databricks -P p1=v1 -P p2=v2',
    );

    expect(wrapper.html()).toContain('Git Commit');
    expect(wrapper.html()).toContain('Entry Point');
    expect(wrapper.html()).toContain('Duration');
    expect(wrapper.html()).toContain('Parent Run');
    expect(wrapper.html()).toContain('Job Output');
  });

  test('state: showNoteEditor false/true -> edit button shown/hidden', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <RunView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    ).find(RunView);

    expect(wrapper.html()).toContain('edit-description-button');
    const runViewInstance = wrapper.find(RunViewImpl).instance();
    runViewInstance.setState({ showNoteEditor: true });
    expect(wrapper.html()).not.toContain('edit-description-button');
  });

  test('should set showRunRenameModal when Rename menu item is clicked', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <RunView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );

    expect(wrapper.find(RunViewImpl).instance().state.showRunRenameModal).toBe(false);
    wrapper
      .find("[data-test-id='overflow-menu-trigger']")
      .at(0)
      .simulate('click');
    wrapper
      .find('[data-test-id="overflow-rename-button"]')
      .hostNodes()
      .simulate('click');
    expect(wrapper.find(RunViewImpl).instance().state.showRunRenameModal).toBe(true);
  });

  test('Requests metric history for run metrics', () => {
    const store = mockStore({
      ...minimalStoreRaw,
      entities: {
        ...minimalStoreRaw.entities,
        latestMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            m1: Metric.fromJs({ key: 'm1', value: 1 }),
            m2: Metric.fromJs({ key: 'm2', value: 2 }),
          },
        },
      },
    });

    mountWithIntl(
      <Provider store={store}>
        <BrowserRouter>
          <RunView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );

    const actions = store.getActions();
    ['m1', 'm2'].forEach((metricKey) => {
      expect(
        actions.find(
          (action) =>
            action.type === pending(GET_METRIC_HISTORY_API) &&
            action.meta.key === metricKey &&
            action.meta.runUuid === 'uuid-1234-5678-9012',
        ),
      ).toBeDefined();
    });
  });

  test('Displays latest, min and max metrics', () => {
    const store = mockStore({
      ...minimalStoreRaw,
      entities: {
        ...minimalStoreRaw.entities,
        latestMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            m1: Metric.fromJs({ key: 'm1', value: 2 }),
            m2: Metric.fromJs({ key: 'm2', value: 3 }),
          },
        },
        minMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            m1: Metric.fromJs({ key: 'm1', value: 1 }),
            m2: Metric.fromJs({ key: 'm2', value: 3 }),
          },
        },
        maxMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            m1: Metric.fromJs({ key: 'm1', value: 5 }),
            m2: Metric.fromJs({ key: 'm2', value: 4 }),
          },
        },
      },
    });

    wrapper = mountWithIntl(
      <Provider store={store}>
        <BrowserRouter>
          <RunView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    ).find(RunView);

    const collapsible = wrapper.find('CollapsibleSection[data-test-id="run-metrics-section"]');
    const table = collapsible.props().children;
    const { columns, values } = table.props;

    expect(columns.length).toBe(4);
    expect(values.length).toBe(2);

    expect(values[0].value.props.children).toBe('2');
    expect(values[0].minValue.props.children).toBe('1');
    expect(values[0].maxValue.props.children).toBe('5');

    expect(values[1].value.props.children).toBe('3');
    expect(values[1].minValue.props.children).toBe('3');
    expect(values[1].maxValue.props.children).toBe('4');
  });
});
