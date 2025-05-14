/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import MetricsSummaryTable from './MetricsSummaryTable';
import { HtmlTableView } from './HtmlTableView';
import { Metric } from '../sdk/MlflowMessages';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';

describe('MetricsSummaryTable', () => {
  let wrapper;
  let minimalProps: any;
  let minimalStore: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);
  const minimalStoreRaw = {
    entities: {
      runInfosByUuid: {
        'uuid-1234-5678-9012': { experimentId: '1' },
        'uuid-1234-5678-9013': { experimentId: '1' },
      },
      latestMetricsByRunUuid: { 'uuid-1234-5678-9012': {} },
      minMetricsByRunUuid: { 'uuid-1234-5678-9012': {} },
      maxMetricsByRunUuid: { 'uuid-1234-5678-9012': {} },
    },
  };
  beforeEach(() => {
    minimalProps = {
      runUuids: ['uuid-1234-5678-9012'],
      runDisplayNames: ['run 0'],
      metricKeys: ['train_loss'],
    };
    minimalStore = mockStore(minimalStoreRaw);
  });
  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <MetricsSummaryTable {...minimalProps} />
        </MemoryRouter>
      </Provider>,
    );
    expect(wrapper.find(MetricsSummaryTable).length).toBe(1);
  });
  test('should render metric summary values', () => {
    const store = mockStore({
      entities: {
        ...minimalStoreRaw.entities,
        latestMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 1, step: 11 }),
          },
        },
        minMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 2, step: 12 }),
          },
        },
        maxMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 3, step: 13 }),
          },
        },
      },
    });
    wrapper = mountWithIntl(
      <Provider store={store}>
        <MemoryRouter>
          <MetricsSummaryTable {...minimalProps} />
        </MemoryRouter>
      </Provider>,
    );
    const table = wrapper.find(HtmlTableView);
    expect(table.length).toBe(1);
    const { columns, values } = table.props();
    expect(columns.length).toBe(4);
    expect(values.length).toBe(1);
    const html = wrapper.html();
    expect(html).toContain('train_loss');
    expect(html).not.toContain('run 0');
    for (let i = 1; i < 4; ++i) {
      expect(html).toContain(`${i} (step=${10 + i})`);
    }
  });
  test('should render multiple metric values for single run', () => {
    const store = mockStore({
      entities: {
        ...minimalStoreRaw.entities,
        latestMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 1, step: 11 }),
            val_loss: (Metric as any).fromJs({ key: 'val_loss', value: 2, step: 12 }),
          },
        },
        minMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 3, step: 13 }),
            val_loss: (Metric as any).fromJs({ key: 'val_loss', value: 4, step: 14 }),
          },
        },
        maxMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 5, step: 15 }),
            val_loss: (Metric as any).fromJs({ key: 'val_loss', value: 6, step: 16 }),
          },
        },
      },
    });
    const props = {
      ...minimalProps,
      metricKeys: ['train_loss', 'val_loss'],
    };
    wrapper = mountWithIntl(
      <Provider store={store}>
        <MemoryRouter>
          <MetricsSummaryTable {...props} />
        </MemoryRouter>
      </Provider>,
    );
    const table = wrapper.find(HtmlTableView);
    expect(table.length).toBe(1);
    const { columns, values } = table.props();
    expect(columns.length).toBe(4);
    expect(values.length).toBe(2);
    const html = wrapper.html();
    expect(html).toContain('train_loss');
    expect(html).toContain('val_loss');
    expect(html).not.toContain('run 0');
    for (let i = 1; i < 7; ++i) {
      expect(html).toContain(`${i} (step=${10 + i})`);
    }
  });
  test('should render single metric for multiple runs', () => {
    const store = mockStore({
      entities: {
        ...minimalStoreRaw.entities,
        latestMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 1, step: 11 }),
          },
          'uuid-1234-5678-9013': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 2, step: 12 }),
          },
        },
        minMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 3, step: 13 }),
          },
          'uuid-1234-5678-9013': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 4, step: 14 }),
          },
        },
        maxMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 5, step: 15 }),
          },
          'uuid-1234-5678-9013': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 6, step: 16 }),
          },
        },
      },
    });
    const props = {
      runUuids: ['uuid-1234-5678-9012', 'uuid-1234-5678-9013'],
      runDisplayNames: ['run 0', 'run 1'],
      metricKeys: ['train_loss'],
    };
    wrapper = mountWithIntl(
      <Provider store={store}>
        <MemoryRouter>
          <MetricsSummaryTable {...props} />
        </MemoryRouter>
      </Provider>,
    );
    const table = wrapper.find(HtmlTableView);
    expect(table.length).toBe(1);
    const { columns, values } = table.props();
    expect(columns.length).toBe(4);
    expect(values.length).toBe(2);
    const html = wrapper.html();
    expect(html).toContain('train_loss');
    expect(html).toContain('run 0');
    expect(html).toContain('run 1');
    expect(html).toContain('experiments/1/runs/uuid-1234-5678-9012');
    expect(html).toContain('experiments/1/runs/uuid-1234-5678-9013');
    for (let i = 1; i < 7; ++i) {
      expect(html).toContain(`${i} (step=${10 + i})`);
    }
  });
  test('should render multiple metrics for multiple runs', () => {
    const store = mockStore({
      entities: {
        ...minimalStoreRaw.entities,
        latestMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 1, step: 11 }),
            val_loss: (Metric as any).fromJs({ key: 'val_loss', value: 2, step: 12 }),
          },
          'uuid-1234-5678-9013': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 3, step: 13 }),
            val_loss: (Metric as any).fromJs({ key: 'val_loss', value: 4, step: 14 }),
          },
        },
        minMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 5, step: 15 }),
            val_loss: (Metric as any).fromJs({ key: 'val_loss', value: 6, step: 16 }),
          },
          'uuid-1234-5678-9013': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 7, step: 17 }),
            val_loss: (Metric as any).fromJs({ key: 'val_loss', value: 8, step: 18 }),
          },
        },
        maxMetricsByRunUuid: {
          'uuid-1234-5678-9012': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 9, step: 19 }),
            val_loss: (Metric as any).fromJs({ key: 'val_loss', value: 10, step: 20 }),
          },
          'uuid-1234-5678-9013': {
            train_loss: (Metric as any).fromJs({ key: 'train_loss', value: 11, step: 21 }),
            val_loss: (Metric as any).fromJs({ key: 'val_loss', value: 12, step: 22 }),
          },
        },
      },
    });
    const props = {
      runUuids: ['uuid-1234-5678-9012', 'uuid-1234-5678-9013'],
      runDisplayNames: ['run 0', 'run 1'],
      metricKeys: ['train_loss', 'val_loss'],
    };
    wrapper = mountWithIntl(
      <Provider store={store}>
        <MemoryRouter>
          <MetricsSummaryTable {...props} />
        </MemoryRouter>
      </Provider>,
    );
    const tables = wrapper.find(HtmlTableView);
    expect(tables.length).toBe(2);
    let { columns, values } = tables.at(0).props();
    expect(columns.length).toBe(4);
    expect(values.length).toBe(2);
    ({ columns, values } = tables.at(1).props());
    expect(columns.length).toBe(4);
    expect(values.length).toBe(2);
    const html = wrapper.html();
    expect(html).toContain('train_loss');
    expect(html).toContain('val_loss');
    expect(html).toContain('run 0');
    expect(html).toContain('run 1');
    expect(html).toContain('experiments/1/runs/uuid-1234-5678-9012');
    expect(html).toContain('experiments/1/runs/uuid-1234-5678-9013');
    for (let i = 1; i < 13; ++i) {
      expect(html).toContain(`${i} (step=${10 + i})`);
    }
  });
  test('should not render table when no metrics', () => {
    const props = {
      runUuids: ['uuid-1234-5678-9012'],
      runDisplayNames: ['run 0'],
      metricKeys: [],
    };
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <MetricsSummaryTable {...props} />
        </MemoryRouter>
      </Provider>,
    );
    const table = wrapper.find(HtmlTableView);
    expect(table.length).toBe(0);
  });
  test('should not render table when no metrics with multiple runs', () => {
    const props = {
      runUuids: ['uuid-1234-5678-9012', 'uuid-1234-5678-9013'],
      runDisplayNames: ['run 0', 'run 1'],
      metricKeys: [],
    };
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <MetricsSummaryTable {...props} />
        </MemoryRouter>
      </Provider>,
    );
    const table = wrapper.find(HtmlTableView);
    expect(table.length).toBe(0);
  });
});
