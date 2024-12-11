/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import thunk from 'redux-thunk';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { mount } from 'enzyme';
import { Metric, Param } from '../sdk/MlflowMessages';
import { CompareRunScatter, CompareRunScatterImpl } from './CompareRunScatter';
import { ArtifactNode } from '../utils/ArtifactUtils';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';

describe('CompareRunScatter', () => {
  let wrapper;
  let minimalProps: any;
  let minimalStoreRaw: any;
  let minimalStore: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);
  beforeEach(() => {
    minimalProps = {
      runUuids: ['uuid-1234-5678-9012', 'uuid-1234-5678-9013'],
      runDisplayNames: ['Run 9012', 'Run 9013'],
    };
    minimalStoreRaw = {
      entities: {
        runInfosByUuid: {
          'uuid-1234-5678-9012': {
            runUuid: 'uuid-1234-5678-9012',
            experimentId: '12345',
            userId: 'me@me.com',
            status: 'RUNNING',
            startTime: 12345678990,
            endTime: 12345678999,
            artifactUri: 'dbfs:/databricks/abc/uuid-1234-5678-9012',
            lifecycleStage: 'active',
          },
          'uuid-1234-5678-9013': {
            runUuid: 'uuid-1234-5678-9013',
            experimentId: '12345',
            userId: 'me@me.com',
            status: 'RUNNING',
            startTime: 12345678990,
            endTime: 12345678999,
            artifactUri: 'dbfs:/databricks/abc/uuid-1234-5678-9013',
            lifecycleStage: 'active',
          },
        },
        artifactsByRunUuid: {
          // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
          'uuid-1234-5678-9012': new ArtifactNode(true),
          // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
          'uuid-1234-5678-9013': new ArtifactNode(true),
        },
        experimentsById: {
          12345: {
            experimentId: 12345,
            name: 'my experiment',
            artifactLocation: 'dbfs:/databricks/abc',
            lifecycleStage: 'active',
            lastUpdateTime: 12345678999,
            creationTime: 12345678900,
            tags: [],
          },
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
        <MemoryRouter>
          <CompareRunScatter {...minimalProps} />
        </MemoryRouter>
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
            p1: (Param as any).fromJs({ key: 'p1', value: 'v11' }),
            p2: (Param as any).fromJs({ key: 'p2', value: 'v12' }),
            p3: (Param as any).fromJs({ key: 'p3', value: 'v13' }),
          },
          'uuid-1234-5678-9013': {
            p1: (Param as any).fromJs({ key: 'p1', value: 'v21' }),
            p2: (Param as any).fromJs({ key: 'p2', value: 'v22' }),
          },
        },
        latestMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            m1: (Metric as any).fromJs({ key: 'm1', value: 1.1 }),
            m3: (Metric as any).fromJs({ key: 'm3', value: 1.3 }),
          },
          'uuid-1234-5678-9013': {
            m1: (Metric as any).fromJs({ key: 'm1', value: 2.1 }),
            m2: (Metric as any).fromJs({ key: 'm2', value: 2.2 }),
          },
        },
      },
    });
    wrapper = mountWithIntl(
      <Provider store={store}>
        <MemoryRouter>
          <CompareRunScatter {...minimalProps} />
        </MemoryRouter>
      </Provider>,
    ).find(CompareRunScatter);
    const instance = wrapper.find(CompareRunScatterImpl).instance();
    expect(instance.state.disabled).toBe(false);
    // getPlotlyTooltip
    expect(instance.getPlotlyTooltip(0)).toEqual(
      '<b>Run 9012</b><br>p1: v11<br>p2: v12<br>p3: v13<br><br>m1: 1.1<br>m3: 1.3<br>',
    );
    expect(instance.getPlotlyTooltip(1)).toEqual('<b>Run 9013</b><br>p1: v21<br>p2: v22<br><br>m1: 2.1<br>m2: 2.2<br>');
  });
});
