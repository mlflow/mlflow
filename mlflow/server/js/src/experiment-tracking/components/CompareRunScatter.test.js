import React from 'react';
import thunk from 'redux-thunk';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { mount } from 'enzyme';
import { Experiment, Metric, Param, RunInfo } from '../sdk/MlflowMessages';
import { CompareRunScatter, CompareRunScatterImpl } from './CompareRunScatter';
import { ArtifactNode } from '../utils/ArtifactUtils';
import { mountWithIntl } from '../../common/utils/TestUtils';

describe('CompareRunScatter', () => {
  let wrapper;
  let minimalProps;
  let minimalStoreRaw;
  let minimalStore;

  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    minimalProps = {
      runUuids: ['uuid-1234-5678-9012', 'uuid-1234-5678-9013'],
      runDisplayNames: ['Run 9012', 'Run 9013'],
    };
    minimalStoreRaw = {
      entities: {
        runInfosByUuid: {
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
          'uuid-1234-5678-9013': RunInfo.fromJs({
            run_uuid: 'uuid-1234-5678-9013',
            experiment_id: '12345',
            user_id: 'me@me.com',
            status: 'RUNNING',
            start_time: 12345678990,
            end_time: 12345678999,
            artifact_uri: 'dbfs:/databricks/abc/uuid-1234-5678-9013',
            lifecycle_stage: 'active',
          }),
        },
        artifactsByRunUuid: {
          'uuid-1234-5678-9012': new ArtifactNode(true),
          'uuid-1234-5678-9013': new ArtifactNode(true),
        },
        experimentsById: {
          12345: Experiment.fromJs({
            experiment_id: 12345,
            name: 'my experiment',
            artifact_location: 'dbfs:/databricks/abc',
            lifecycle_stage: 'active',
            last_update_time: 12345678999,
            creation_time: 12345678900,
            tags: [],
          }),
        },
        tagsByRunUuid: {
          'uuid-1234-5678-9012': {},
          'uuid-1234-5678-9013': {},
        },
        paramsByRunUuid: {
          'uuid-1234-5678-9012': {},
          'uuid-1234-5678-9013': {},
        },
        latestMetricsByRunUuid: {
          'uuid-1234-5678-9012': {},
          'uuid-1234-5678-9013': {},
        },
        artifactRootUriByRunUuid: {
          'uuid-1234-5678-9012': 'root/uri',
          'uuid-1234-5678-9013': 'root/uri2',
        },
      },
      apis: {},
    };
    minimalStore = mockStore(minimalStoreRaw);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <CompareRunScatter {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    ).find(CompareRunScatter);
    expect(wrapper.find(CompareRunScatterImpl).length).toBe(1);
    const instance = wrapper.find(CompareRunScatterImpl).instance();
    // nothing is rendered
    expect(instance.state.disabled).toBe(true);
    expect(wrapper.find({ label: 'Parameter' }).length).toBe(0);
  });

  test('with params and metrics - select rendering, getPlotlyTooltip', () => {
    const store = mockStore({
      ...minimalStoreRaw,
      entities: {
        ...minimalStoreRaw.entities,
        paramsByRunUuid: {
          'uuid-1234-5678-9012': {
            p1: Param.fromJs({ key: 'p1', value: 'v11' }),
            p2: Param.fromJs({ key: 'p2', value: 'v12' }),
            p3: Param.fromJs({ key: 'p3', value: 'v13' }),
          },
          'uuid-1234-5678-9013': {
            p1: Param.fromJs({ key: 'p1', value: 'v21' }),
            p2: Param.fromJs({ key: 'p2', value: 'v22' }),
          },
        },
        latestMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            m1: Metric.fromJs({ key: 'm1', value: 1.1 }),
            m3: Metric.fromJs({ key: 'm3', value: 1.3 }),
          },
          'uuid-1234-5678-9013': {
            m1: Metric.fromJs({ key: 'm1', value: 2.1 }),
            m2: Metric.fromJs({ key: 'm2', value: 2.2 }),
          },
        },
      },
    });
    wrapper = mountWithIntl(
      <Provider store={store}>
        <BrowserRouter>
          <CompareRunScatter {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    ).find(CompareRunScatter);

    const instance = wrapper.find(CompareRunScatterImpl).instance();
    expect(instance.state.disabled).toBe(false);
    // params show up in select
    expect(wrapper.find({ label: 'Parameter' }).length).toBe(2);
    expect(wrapper.find({ value: 'param-p1' }).length).toBe(3); // default select, option x, option y
    expect(wrapper.find({ value: 'param-p2' }).length).toBe(2); // option x, option y
    expect(wrapper.find({ value: 'param-p3' }).length).toBe(2); // option x, option y
    expect(wrapper.find({ label: 'Metric' }).length).toBe(2);
    // metrics show up in select
    expect(wrapper.find({ label: 'Parameter' }).length).toBe(2);
    expect(wrapper.find({ value: 'metric-m1' }).length).toBe(3); // default select, option x, option y
    expect(wrapper.find({ value: 'metric-m2' }).length).toBe(2); // option x, option y
    expect(wrapper.find({ value: 'metric-m3' }).length).toBe(2); // option x, option y
    expect(wrapper.find({ label: 'Metric' }).length).toBe(2);

    // getPlotlyTooltip
    expect(instance.getPlotlyTooltip(0)).toEqual(
      '<b>Run 9012</b><br>p1: v11<br>p2: v12<br>p3: v13<br><br>m1: 1.1<br>m3: 1.3<br>',
    );
    expect(instance.getPlotlyTooltip(1)).toEqual(
      '<b>Run 9013</b><br>p1: v21<br>p2: v22<br><br>m1: 2.1<br>m2: 2.2<br>',
    );
  });
});
